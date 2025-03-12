import os
import sys
import shutil
import warnings
from abc import ABC, abstractmethod
import platform

import numpy as np
import scipy.sparse as sp

from cvxpygen.utils import write_file, read_write_file, write_struct_prot, write_struct_def, \
    write_vec_prot, write_vec_def, multiple_replace, cut_from_expr, \
    write_description, type_to_cast, write_mat_def, write_L_def, ones, zeros, write_canonicalize
from cvxpygen.mappings import PrimalVariableInfo, DualVariableInfo, ConstraintInfo, AffineMap, \
    ParameterCanon, WorkspacePointerInfo, UpdatePendingLogic, ParameterUpdateLogic

from cvxpy.reductions.solvers.qp_solvers.osqp_qpif import OSQP
from cvxpy.reductions.solvers.conic_solvers.scs_conif import SCS
from cvxpy.reductions.solvers.conic_solvers.ecos_conif import ECOS
from cvxpy.reductions.solvers.conic_solvers.clarabel_conif import CLARABEL


def get_interface_class(solver_name: str) -> "SolverInterface":
    if platform.system() == 'Windows' and solver_name.upper() == 'CLARABEL':
        raise ValueError(f'Clarabel solver currently unsupported on Windows.')
    mapping = {
        'CLARABEL': (ClarabelInterface, CLARABEL)
    }
    interface = mapping.get(solver_name.upper(), None)
    if interface is None:
        raise ValueError(f'Unsupported solver: {solver_name}.')
    return interface[0], interface[1]


class SolverInterface(ABC):

    def __init__(self, solver_name, n_var, n_eq, n_ineq, indices_obj, indptr_obj, shape_obj,
                 indices_constr, indptr_constr, shape_constr, canon_constants, enable_settings):
        self.solver_name = solver_name
        self.n_var = n_var
        self.n_eq = n_eq
        self.n_ineq = n_ineq
        self.indices_obj = indices_obj
        self.indptr_obj = indptr_obj
        self.shape_obj = shape_obj
        self.indices_constr = indices_constr
        self.indptr_constr = indptr_constr
        self.shape_constr = shape_constr
        self.canon_constants = canon_constants
        self.enable_settings = enable_settings

        self.configure_settings()

    @property
    @abstractmethod
    def canon_p_ids(self):
        pass

    @property
    @abstractmethod
    def canon_p_ids_constr_vec(self):
        pass

    @property
    @abstractmethod
    def stgs_names(self):
        pass

    @property
    @abstractmethod
    def stgs_types(self):
        pass

    @property
    @abstractmethod
    def stgs_defaults(self):
        pass

    @staticmethod
    def ret_prim_func_exists(variable_info: PrimalVariableInfo) -> bool:
        return any(variable_info.sym) or any([s == 1 for s in variable_info.sizes])

    @staticmethod
    def ret_dual_func_exists(dual_variable_info: DualVariableInfo) -> bool:
        return any([s == 1 for s in dual_variable_info.sizes])

    def configure_settings(self) -> None:
        for i, s in enumerate(self.stgs_names):
            if s in self.enable_settings:
                self.stgs_enabled[i] = True
        for s in set(self.enable_settings)-set(self.stgs_names):
            warnings.warn(f'Cannot enable setting {s} for solver {self.solver_name}')

    def get_affine_map(self, p_id, param_prob, constraint_info: ConstraintInfo) -> AffineMap:
        affine_map = AffineMap()

        if p_id == 'P':
            if self.indices_obj is None: # problem is an LP
                return None
            affine_map.mapping = param_prob.reduced_P.reduced_mat
            affine_map.indices = self.indices_obj
            affine_map.shape = (self.n_var, self.n_var)
        elif p_id in ['q', 'c']:
            affine_map.mapping = param_prob.c[:-1]
        elif p_id == 'd':
            affine_map.mapping = param_prob.c[[-1], :]
        elif p_id == 'A':
            affine_map.mapping_rows = constraint_info.mapping_rows_eq[
                constraint_info.mapping_rows_eq < constraint_info.n_data_constr_mat]
            affine_map.shape = (self.n_eq, self.n_var)
        elif p_id == 'G':
            affine_map.mapping_rows = constraint_info.mapping_rows_ineq[
                constraint_info.mapping_rows_ineq < constraint_info.n_data_constr_mat]
            affine_map.shape = (self.n_ineq, self.n_var)
        elif p_id == 'b':
            affine_map.mapping_rows = constraint_info.mapping_rows_eq[
                constraint_info.mapping_rows_eq >= constraint_info.n_data_constr_mat]
            affine_map.shape = (self.n_eq, 1)
        elif p_id == 'h':
            affine_map.mapping_rows = constraint_info.mapping_rows_ineq[
                constraint_info.mapping_rows_ineq >= constraint_info.n_data_constr_mat]
            affine_map.shape = (self.n_ineq, 1)
        else:
            raise ValueError(f'Unknown parameter name: {p_id}.')

        if p_id in ['A', 'b']:
            affine_map.indices = self.indices_constr[affine_map.mapping_rows]
        elif p_id in ['G', 'h']:
            affine_map.indices = self.indices_constr[affine_map.mapping_rows] - self.n_eq

        if p_id in ['A', 'G']:
            affine_map.mapping = param_prob.reduced_A.reduced_mat[affine_map.mapping_rows]
            affine_map.sign = -1

        return affine_map
    
    def augment_vector_parameter(self, p_id, vector_parameter):
        return vector_parameter
    
    def get_problem_data_index(self, reduced_mat):
        if reduced_mat.problem_data_index is None:
            return None, None, None
        else:
            indices, indptr, shape = reduced_mat.problem_data_index
            return indices, indptr, shape

    @property
    def stgs_names_enabled(self):
        return [name for name, enabled in zip(self.stgs_names, self.stgs_enabled) if enabled]

    @property
    def stgs_names_to_type(self):
        return {name: typ for name, typ, enabled in zip(self.stgs_names, self.stgs_types, self.stgs_enabled)
                if enabled}

    @property
    def stgs_names_to_default(self):
        return {name: typ for name, typ, enabled in zip(self.stgs_names, self.stgs_defaults, self.stgs_enabled)
                if enabled}

    @staticmethod
    def check_unsupported_cones(cone_dims: "ConeDims") -> None:
        pass

    @abstractmethod
    def generate_code(self, configuration, code_dir, solver_code_dir, cvxpygen_directory,
                      parameter_canon: ParameterCanon, gradient, prefix) -> None:
        pass

    def declare_workspace(self, f, prefix, parameter_canon) -> None:
        pass

    def define_workspace(self, f, prefix, parameter_canon) -> None:
        pass
    
    def write_gradient_def(f, configuration,
                           variable_info_first, dual_variable_info_first,
                           variable_info_second, dual_variable_info_second,
                           parameter_info, parameter_canon, solver_interface) -> None:
        pass
    
    def write_gradient_prot(f, configuration,
                            variable_info_first, dual_variable_info_first,
                            variable_info_second, dual_variable_info_second,
                            parameter_info, parameter_canon, solver_interface) -> None:
        pass
    
    def write_gradient_workspace_def(f, prefix, parameter_canon) -> None:
        pass

class ClarabelInterface(SolverInterface):
    solver_name = 'Clarabel'
    solver_type = 'conic'
    canon_p_ids = ['P', 'q', 'd', 'A', 'b']
    canon_p_ids_constr_vec = ['b']
    gradient_supported = False

    # header and source files
    header_files = ['<Clarabel>']
    cmake_headers, cmake_sources = [], []

    # preconditioning of problem data happening in-memory
    inmemory_preconditioning = True

    # workspace
    ws_statically_allocated_in_solver_code = False
    ws_ptrs = WorkspacePointerInfo(
        objective_value = 'solution.obj_val',
        iterations = 'solution.iterations',
        status = 'solution.status',
        primal_residual = 'solution.r_prim',
        dual_residual = 'solution.r_dual',
        primal_solution = 'solution.x',
        dual_solution = 'solution.{dual_var_name}',
        settings = 'settings.{setting_name}'
    )

    # solution vectors statically allocated
    sol_statically_allocated = False

    # solver status as integer vs. string
    status_is_int = True

    # float and integer types
    numeric_types = {'float': 'ClarabelFloat', 'int': 'uintptr_t'}
    
    # solver settings
    stgs_dynamically_allocated = True
    stgs_requires_extra_struct_type = True
    stgs_direct_write_ptr = None
    stgs_reset_function = None
    # NOTE: still missing options 'direct_solve_method' and 'chordal_decomposition_merge_method' (string inputs) and 
    # 'static_regularization_proportional' (uses epsilon of float32/float64)
    stgs_names = [
        # main algorithm settings
        'max_iter', 'time_limit', 'verbose', 'max_step_fraction',
        # full accuracy settings
        'tol_gap_abs', 'tol_gap_rel', 'tol_feas', 'tol_infeas_abs', 'tol_infeas_rel', 'tol_ktratio',
        # reduced accuracy settings
        'reduced_tol_gap_abs', 'reduced_tol_gap_rel', 'reduced_tol_feas', 'reduced_tol_infeas_abs', 'reduced_tol_infeas_rel', 'reduced_tol_ktratio',
        # data equilibration settings
        'equilibrate_enable', 'equilibrate_max_iter', 'equilibrate_min_scaling', 'equilibrate_max_scaling',
        # step size settings
        'linesearch_backtrack_step', 'min_switch_step_length', 'min_terminate_step_length',
        # linear solver settings
        'direct_kkt_solver', 'static_regularization_enable', 'static_regularization_constant', 'dynamic_regularization_enable', 'dynamic_regularization_eps', 'dynamic_regularization_delta',
        'iterative_refinement_enable', 'iterative_refinement_reltol', 'iterative_refinement_abstol', 'iterative_refinement_max_iter', 'iterative_refinement_stop_ratio',
        # preprocessing settings
        'presolve_enable', 
    ]
    stgs_translation = "{}"
    stgs_types = ['cpg_int', 'cpg_float', 'cpg_int', 'cpg_float',
                  'cpg_float', 'cpg_float', 'cpg_float', 'cpg_float', 'cpg_float', 'cpg_float',
                  'cpg_float', 'cpg_float', 'cpg_float', 'cpg_float', 'cpg_float', 'cpg_float',
                  'cpg_int', 'cpg_int', 'cpg_float', 'cpg_float',
                  'cpg_float', 'cpg_float', 'cpg_float',
                  'cpg_int', 'cpg_int', 'cpg_float', 'cpg_int', 'cpg_float', 'cpg_float',
                  'cpg_int', 'cpg_float', 'cpg_float', 'cpg_int', 'cpg_float', 'cpg_int'
                  ]
    stgs_enabled = [True] * len(stgs_names)
    stgs_defaults = ['50', '1e6', '1', '0.99',
                     '1e-8', '1e-8', '1e-8', '1e-8', '1e-8', '1e-6',
                     '5e-5', '5e-5', '1e-4', '5e-5', '5e-5', '1e-4',
                     '1', '10', '1e-4', '1e4',
                     '0.8', '1e-1', '1e-4',
                     '1', '1', '1e-8', '1', '1e-13', '2e-7',
                     '1', '1e-12', '1e-12', '10', '5.0',
                     '1']

    # dual variables split into two vectors
    dual_var_split = False
    dual_var_names = ['z']

    # docu
    docu = 'https://oxfordcontrol.github.io/ClarabelDocs/'

    def __init__(self, data, p_prob, enable_settings):
        n_var = p_prob.x.size
        n_eq = data['A'].shape[0]
        n_ineq = 0

        indices_obj, indptr_obj, shape_obj = self.get_problem_data_index(p_prob.reduced_P)
        indices_constr, indptr_constr, shape_constr = self.get_problem_data_index(p_prob.reduced_A)

        canon_constants = {'n': n_var, 'm': n_eq,
                           'cone_dims_zero': p_prob.cone_dims.zero,
                           'cone_dims_nonneg': p_prob.cone_dims.nonneg,
                           'cone_dims_exp': p_prob.cone_dims.exp,
                           'cone_dims_soc': np.array(p_prob.cone_dims.soc),
                           'cone_dims_psd': np.array(p_prob.cone_dims.psd),
                           'cone_dims_p3d': np.array(p_prob.cone_dims.p3d)}
        
        canon_constants['n_cone_types'] = int(p_prob.cone_dims.zero > 0) + \
            int(p_prob.cone_dims.nonneg > 0) + \
            int(p_prob.cone_dims.exp > 0) + \
            len(p_prob.cone_dims.soc) + \
            len(p_prob.cone_dims.psd) + \
            len(p_prob.cone_dims.p3d)
        
        canon_constants['cd_to_t'] = {
            'cone_dims_zero': 'ClarabelZeroConeT',
            'cone_dims_nonneg': 'ClarabelNonnegativeConeT',
            'cone_dims_exp': 'ClarabelExponentialConeT',
            'cone_dims_soc': 'ClarabelSecondOrderConeT',
            'cone_dims_psd': 'ClarabelPSDTriangleConeT',
            'cone_dims_p3d': 'ClarabelPowerConeT'
        }

        # catch LP case (hack, until Clarabel permits passing a zero pointer as &P)
        if indices_obj is None:
            extra_condition = '1'
            P_p = f'(cpg_int[]){{{{ {", ".join(["0"]*(n_var+1))} }}}}'
            P_i = '0'
            P_x = '0'
            update_after_init = ['A', 'q', 'b']
        else:
            extra_condition = '!{prefix}solver'
            P_p = '{prefix}Canon_Params_conditioning.P->p'
            P_i = '{prefix}Canon_Params_conditioning.P->i'
            P_x = '{prefix}Canon_Params_conditioning.P->x'
            update_after_init = ['P', 'A', 'q', 'b']

        self.parameter_update_structure = {
            'init': ParameterUpdateLogic(
                update_pending_logic=UpdatePendingLogic([], extra_condition=extra_condition, functions_if_false=update_after_init),
                function_call= \
                    f'{{prefix}}copy_all();\n'
                    f'    clarabel_CscMatrix_init(&{{prefix}}P, {canon_constants["n"]}, {canon_constants["n"]}, {P_p}, {P_i}, {P_x});\n'
                    f'    clarabel_CscMatrix_init(&{{prefix}}A, {canon_constants["m"]}, {canon_constants["n"]}, {{prefix}}Canon_Params_conditioning.A->p, {{prefix}}Canon_Params_conditioning.A->i, {{prefix}}Canon_Params_conditioning.A->x);\n' \
                    f'    {{prefix}}settings = clarabel_DefaultSettings_default()'
            ),
            'A': ParameterUpdateLogic(
                update_pending_logic = UpdatePendingLogic(['A']),
                function_call = f'{{prefix}}copy_A();\n      clarabel_CscMatrix_init(&{{prefix}}A, {canon_constants["m"]}, {canon_constants["n"]}, {{prefix}}Canon_Params_conditioning.A->p, {{prefix}}Canon_Params_conditioning.A->i, {{prefix}}Canon_Params_conditioning.A->x)'
            ),
            'q': ParameterUpdateLogic(
                update_pending_logic = UpdatePendingLogic(['q']),
                function_call = f'{{prefix}}copy_q()'
            ),
            'b': ParameterUpdateLogic(
                update_pending_logic = UpdatePendingLogic(['b']),
                function_call = f'{{prefix}}copy_b()'
            ),
        }
        
        if indices_obj is not None:
            self.parameter_update_structure['P'] = ParameterUpdateLogic(
                update_pending_logic = UpdatePendingLogic(['P']),
                function_call = f'{{prefix}}copy_P();\n      clarabel_CscMatrix_init(&{{prefix}}P, {canon_constants["n"]}, {canon_constants["n"]}, {P_p}, {P_i}, {P_x})'
            )

        self.solve_function_call = \
            f'{{prefix}}solver = clarabel_DefaultSolver_new(&{{prefix}}P, {{prefix}}Canon_Params_conditioning.q, &{{prefix}}A, {{prefix}}Canon_Params_conditioning.b, {canon_constants["n_cone_types"]}, {{prefix}}cones, &{{prefix}}settings);\n' \
            f'  clarabel_DefaultSolver_solve({{prefix}}solver);\n' \
            f'  {{prefix}}solution = clarabel_DefaultSolver_solution({{prefix}}solver)'

        super().__init__(self.solver_name, n_var, n_eq, n_ineq, indices_obj, indptr_obj, shape_obj,
                         indices_constr, indptr_constr, shape_constr, canon_constants, enable_settings)

    @staticmethod
    def ret_prim_func_exists(variable_info: PrimalVariableInfo) -> bool:
        return True

    @staticmethod
    def ret_dual_func_exists(dual_variable_info: DualVariableInfo) -> bool:
        return True

    def generate_code(self, configuration, code_dir, solver_code_dir, cvxpygen_directory,
                    parameter_canon: ParameterCanon, gradient, prefix) -> None:

        # check if sdp cones are present
        is_sdp = len(self.canon_constants['cone_dims_psd']) > 0
        if is_sdp:
            sys.stdout.write('WARNING: You are generating code for an SDP with Clarabel, which requires BLAS and LAPACK within Rust-C wrappers - expect large compilation time and binary size.\n')

        # copy sources
        if os.path.isdir(solver_code_dir):
            shutil.rmtree(solver_code_dir)
        os.mkdir(solver_code_dir)
        dirs_to_copy = ['rust_wrapper', 'include', 'Clarabel.rs']
        for dtc in dirs_to_copy:
            shutil.copytree(os.path.join(cvxpygen_directory, 'solvers', 'Clarabel.cpp', dtc),
                            os.path.join(solver_code_dir, dtc))
        files_to_copy = ['CMakeLists.txt', 'LICENSE.md']
        for fl in files_to_copy:
            shutil.copyfile(os.path.join(cvxpygen_directory, 'solvers', 'Clarabel.cpp', fl),
                            os.path.join(solver_code_dir, fl))
        shutil.copy(os.path.join(cvxpygen_directory, 'template', 'LICENSE'), code_dir)

        # adjust top-level CMakeLists.txt
        with open(os.path.join(code_dir, 'c', 'CMakeLists.txt'), 'a') as f:
            if is_sdp:
                f.write('\nfind_package(BLAS REQUIRED)')
                f.write('\nfind_package(LAPACK REQUIRED)')
                link_libraries = 'libclarabel_c_static ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES}'
            else:
                link_libraries = 'libclarabel_c_static'
            f.write(f'\ntarget_link_libraries(cpg_example PRIVATE {link_libraries})')
            f.write(f'\ntarget_link_libraries(cpg PRIVATE {link_libraries})\n')

        # remove examples target from Clarabel.cpp/CMakeLists.txt and set build type to Release
        replacements = [
            ('add_subdirectory(examples)', '# add_subdirectory(examples)'),
            ('set(CMAKE_C_STANDARD_REQUIRED True)', 'set(CMAKE_C_STANDARD_REQUIRED True)\n\n# set build type to Release\nset(CMAKE_BUILD_TYPE Release)')
        ]
        read_write_file(os.path.join(code_dir, 'c', 'solver_code', 'CMakeLists.txt'),
                        lambda x: multiple_replace(x, replacements))

        # add sdp flag
        if is_sdp:
            read_write_file(os.path.join(code_dir, 'c', 'solver_code', 'CMakeLists.txt'),
                            lambda x: x.replace('set(CLARABEL_FEATURE_SDP "none"', 'set(CLARABEL_FEATURE_SDP "sdp-openblas"'))

        # adjust Clarabel.cpp/rust_wrapper/CMakeLists.txt
        replacements = [
            ('${CMAKE_SOURCE_DIR}/', '${CMAKE_SOURCE_DIR}/solver_code/'),
            ('/libclarabel_c.lib', '/clarabel_c.lib'),  # until fixed on Clarabel side
            (
                'set(clarabel_c_output_directory "${CMAKE_SOURCE_DIR}/solver_code/rust_wrapper/target/release")',
                'if (ARM64)\n'
                '        message(STATUS "ARM64 detected")\n'
                '        set(clarabel_c_output_directory "${CMAKE_SOURCE_DIR}/solver_code/rust_wrapper/target/aarch64-apple-darwin/release")\n'
                '    else()\n'
                '        set(clarabel_c_output_directory "${CMAKE_SOURCE_DIR}/solver_code/rust_wrapper/target/release")\n'
                '    endif()'
            ),
            (
                '# Add the cargo project as a custom target',
                '# Add the cargo project as a custom target\n'
                'if(ARM64)\n'
                '   set(clarabel_c_build_flags "${clarabel_c_build_flags};--target;aarch64-apple-darwin")\n'
                'endif()'
            )
        ]
        read_write_file(os.path.join(code_dir, 'c', 'solver_code', 'rust_wrapper', 'CMakeLists.txt'),
                        lambda x: multiple_replace(x, replacements))

        # adjust Clarabel
        read_write_file(os.path.join(code_dir, 'c', 'solver_code', 'include', 'Clarabel'),
                        lambda x: x.replace('cpp/', 'c/'))

        # adjust setup.py
        release_dir = "'aarch64-apple-darwin/release'" if platform.system() == "Darwin" and platform.machine() == "arm64" else "'release'"
        read_write_file(os.path.join(code_dir, 'setup.py'),
                        lambda x: x.replace("extra_objects=[cpg_lib])",
                                            f"extra_objects=[cpg_lib, os.path.join(cpg_dir, 'solver_code', 'rust_wrapper', 'target', {release_dir}, 'libclarabel_c.a')])"))

    
    def declare_workspace(self, f, prefix, parameter_canon) -> None:
        f.write('\n// Clarabel workspace\n')
        f.write(f'extern ClarabelCscMatrix {prefix}P;\n')
        f.write(f'extern ClarabelCscMatrix {prefix}A;\n')
        f.write(f'extern ClarabelSupportedConeT {prefix}cones[{self.canon_constants["n_cone_types"]}];\n')
        f.write(f'extern ClarabelDefaultSettings {prefix}settings;\n')
        f.write(f'extern ClarabelDefaultSolver *{prefix}solver;\n')
        f.write(f'extern ClarabelDefaultSolution {prefix}solution;\n')

    def define_workspace(self, f, prefix, parameter_canon) -> None:
        f.write('\n// Clarabel workspace\n')
        f.write(f'ClarabelCscMatrix {prefix}P;\n')
        f.write(f'ClarabelCscMatrix {prefix}A;\n')
        cone_str_list = []
        for cd in ['cone_dims_zero', 'cone_dims_nonneg']:
            if self.canon_constants[cd] > 0:
                cone_str_list.append(f'{self.canon_constants["cd_to_t"][cd]}({self.canon_constants[cd]})')
        cone_str_list.extend(['ClarabelExponentialConeT()'] * self.canon_constants['cone_dims_exp'])
        for cd in ['cone_dims_soc', 'cone_dims_psd', 'cone_dims_p3d']:
            for l in self.canon_constants[cd]:
                cone_str_list.append(f'{self.canon_constants["cd_to_t"][cd]}({l})')
        f.write(f'ClarabelSupportedConeT {prefix}cones[{self.canon_constants["n_cone_types"]}] = {{ {", ".join(cone_str_list)} }};\n')
        f.write(f'ClarabelDefaultSettings {prefix}settings;\n')
        f.write(f'ClarabelDefaultSolver *{prefix}solver = 0;\n')
        f.write(f'ClarabelDefaultSolution {prefix}solution;\n')
