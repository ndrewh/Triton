
#ifdef TRITON_BOOLECTOR_INTERFACE

#define bitwuzla_set_option boolector_set_opt
#define BITWUZLA_OPT_PRODUCE_MODELS BTOR_OPT_MODEL_GEN
#define BITWUZLA_OPT_INCREMENTAL BTOR_OPT_INCREMENTAL
#define BITWUZLA_OPT_RW_EXTRACT_ARITH BTOR_OPT_RW_ZERO_LOWER_SLICE
#define BITWUZLA_OPT_CHECK_MODEL BTOR_OPT_CHK_MODEL
#define BITWUZLA_OPT_PRODUCE_UNSAT_CORES BTOR_OPT_UCOPT

#define bitwuzla_delete boolector_delete
#define bitwuzla_assert boolector_assert
#define bitwuzla_assume boolector_assume

#define bitwuzla_check_sat boolector_sat

#define bitwuzla_term_array_get_index_sort(x) boolector_bitvec_sort(_bzla, boolector_get_index_width(_bzla, x))
#define bitwuzla_mk_bv_value_uint64 boolector_mk_bv_value_uint64

#define bitwuzla_get_const_bv_value boolector_get_bits
#define bitwuzla_get_bv_value boolector_bv_assignment
#define bitwuzla_get_value boolector_get_value
#define bitwuzla_sort_bv_get_size(x) boolector_bitvec_sort_get_width(_bzla, x)
#define TERM_IS_VALUE(x, solver) boolector_is_const(solver, x)
#define TERM_IS_ARRAY(x, solver) boolector_is_array(solver, x)

#define bitwuzla_mk_bv_sort boolector_bitvec_sort
#define bitwuzla_new boolector_new
#define bitwuzla_set_termination_callback boolector_set_term
#define bitwuzla_set_abort_callback boolector_set_abort

#define bitwuzla_simplify boolector_simplify
#define bitwuzla_dump_formula_and_term boolector_dump_formula_and_term
#define bitwuzla_get_output_term boolector_get_output_term
#define bitwuzla_get_num_outputs boolector_get_num_outputs

#define bitwuzla_set_is_array boolector_set_is_array
#define bitwuzla_mk_true boolector_true

#define bitwuzla_mk_array_sort boolector_array_sort
#define bitwuzla_mk_const_array boolector_const_array

#define BITWUZLA_SAT BOOLECTOR_SAT
#define BITWUZLA_UNSAT BOOLECTOR_UNSAT
#define BITWUZLA_UNKNOWN BOOLECTOR_UNKNOWN

#else
#define TERM_IS_VALUE(x, solver) bitwuzla_term_is_value(x)
#define TERM_IS_ARRAY(x, solver) bitwuzla_term_is_array(x)

#endif
