//! \file
/*
**  Copyright (C) - Triton
**
**  This program is under the terms of the Apache License 2.0.
*/

#include <fstream>
#include <regex>
#include <string>

#include <triton/astContext.hpp>
#include <triton/bitwuzlaSolver.hpp>
#include <triton/bitwuzlaCachingSolver.hpp>
#include <triton/exceptions.hpp>
#include <triton/solverModel.hpp>
#include <triton/symbolicExpression.hpp>
#include <triton/symbolicVariable.hpp>
#include <triton/tritonTypes.hpp>

// We implement a bitwuzla and boolector solver in this one file.
// For boolector, we simply replace the bitwuzla API names into the boolector API names.
#include <triton/bitwuzlaBoolectorCompat.hpp>

namespace triton {
  namespace engines {
    namespace solver {
      using triton::ast::SharedAbstractNode;

      BitwuzlaCachingSolver::BitwuzlaCachingSolver(bool incremental, bool needs_nonincremental_shadow) : _incremental(incremental) {
        this->_bzlaAst = std::move(TritonToAst(false, true));

        this->timeout = 0;
        this->memoryLimit = 0;

        // Set bitwuzla abort function.
        bitwuzla_set_abort_callback(this->abortCallback);
        _solver = bitwuzla_new();
        bitwuzla_set_option(_solver, BITWUZLA_OPT_PRODUCE_MODELS, 1);
        bitwuzla_set_option(_solver, BITWUZLA_OPT_INCREMENTAL, _incremental);
        bitwuzla_set_option(_solver, BITWUZLA_OPT_RW_EXTRACT_ARITH, 1);
        bitwuzla_set_option(_solver, BITWUZLA_OPT_CHECK_MODEL, 0);
#ifdef TRITON_BOOLECTOR_INTERFACE
        boolector_set_opt(_solver, BTOR_OPT_AUTO_CLEANUP, 1);
#endif
        // bitwuzla_set_option_str(_solver, BITWUZLA_OPT_PP_BETA_REDUCE, "all");

        // if (!_incremental) {
          // bitwuzla_set_option(_solver, BITWUZLA_OPT_PRODUCE_UNSAT_CORES, 1);
        // }

        if (_incremental && needs_nonincremental_shadow) {
          _nonincremental = std::make_unique<BitwuzlaCachingSolver>(BitwuzlaCachingSolver(false));
        }

        // Make it load shit
        // XXX: why is this here? this breaks nonincremental
        // auto res = bitwuzla_check_sat(_solver);
      }
      BitwuzlaCachingSolver::~BitwuzlaCachingSolver() {
        if (_solver) {
          bitwuzla_delete(_solver);
        }
      }
      // move-construct
      BitwuzlaCachingSolver::BitwuzlaCachingSolver(BitwuzlaCachingSolver &&other) {
        this->_bzlaAst = std::move(other._bzlaAst);
        this->timeout = other.timeout;
        this->memoryLimit = other.memoryLimit;
        this->_solver = other._solver;
        this->_incremental = other._incremental;
        this->_nonincremental = std::move(other._nonincremental);
        other._solver = nullptr;
      }
      // move-assign
      BitwuzlaCachingSolver& BitwuzlaCachingSolver::operator=(BitwuzlaCachingSolver &&other) { // move-assign
        this->_bzlaAst = std::move(other._bzlaAst);
        this->timeout = other.timeout;
        this->memoryLimit = other.memoryLimit;
        if (this->_solver) {
          bitwuzla_delete(this->_solver);
        }
        this->_solver = other._solver;
        this->_incremental = other._incremental;
        this->_nonincremental = std::move(other._nonincremental);
        other._solver = nullptr;
        return *this;
      }

      SolverTerm convert(TritonToAst &ast, SolverPtr bzla, const SharedAbstractNode &node) {
        if (node == nullptr)
          throw triton::exceptions::SolverEngine("BitwuzlaCachingSolver::getModels(): Node cannot be null.");

        //
        // Get time of solving start.
        auto start = std::chrono::system_clock::now();

        auto res = ast.convert(node, bzla);

        auto end = std::chrono::system_clock::now();
        auto delta = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        if (delta > 250) {
          std::cout << "convert took " << std::dec << delta << "\n";
        }
        return res;
      }

      void BitwuzlaCachingSolver::add_permanent(const SharedAbstractNode &node) {
        if (node->isLogical() == false)
          throw triton::exceptions::SolverEngine("BitwuzlaCachingSolver::getModels(): Must be a logical node.");
        if (node->evaluate() == 0) {
          node->getContext()->setRepresentationMode(triton::ast::representations::PYTHON_REPRESENTATION);
          node->getContext()->print(std::cout, node.get()) << "\n";
          throw triton::exceptions::SolverEngine("BitwuzlaCachingSolver::add_permanent(): Added false assertion");
        }
        // std::cout << "add_permanent:\n";
        // node->getContext()->print(std::cout, node.get()) << "\n" << std::flush;
        auto converted = convert(_bzlaAst, _solver, node);

        // bitwuzla_term_dump(converted, "smt2", stdout);
        // std::cout << "\n" << std::flush;
        bitwuzla_assert(_solver, converted);

        // NOTE: The fact that nonincremental is always called second is pretty important
        if (_nonincremental) {
          _nonincremental->add_permanent(node);
        }
      }

      void BitwuzlaCachingSolver::substitute(const SharedAbstractNode& varnode, const SharedAbstractNode &new_val) {
          auto converted = convert(_bzlaAst, _solver, new_val);
          _bzlaAst.substitute(varnode, converted, _solver);
          if (_nonincremental) {
            _nonincremental->substitute(varnode, new_val);
          }
      }

      std::vector<std::unordered_map<triton::usize, SolverModel>> BitwuzlaCachingSolver::getModels(const triton::ast::SharedAbstractNode& node,
                                                                                            triton::uint32 limit,
                                                                                            triton::engines::solver::status_e* status,
                                                                                            triton::uint32 timeout,
                                                                                            triton::uint32* solvingTime) {
        // Convert Triton' AST to solver terms.
        if (node->isLogical() == false)
          throw triton::exceptions::SolverEngine("BitwuzlaCachingSolver::getModels(): Must be a logical node.");

        if (!_incremental)
          throw triton::exceptions::SolverEngine("BitwuzlaCachingSolver::getModels(): Cannot be called on non-incremental solver");



        auto assumption = convert(_bzlaAst, _solver, node);

        // This assumption is temporary!!
        bitwuzla_assume(_solver, assumption);

        // Set solving params.
        timeout = timeout == 0 ? this->timeout : timeout;
        SolverParams p(timeout, this->memoryLimit);
        if (timeout || this->memoryLimit) {
          bitwuzla_set_termination_callback(_solver, this->terminateCallback, reinterpret_cast<void*>(&p));
        }
        // bitwuzla_set_option(_solver, BITWUZLA_OPT_PRINT_DIMACS, printSAT ? 1 : 0);

        // Get time of solving start.
        auto start = std::chrono::system_clock::now();

        // Check result.
        fprintf(stdout, "bitwuzla_check_sat getModels: %p\n", _solver);
        auto res = bitwuzla_check_sat(_solver);

        // Write back status.
        if (status) {
          switch (res) {
            case BITWUZLA_SAT:
              *status = triton::engines::solver::SAT;
              break;
            case BITWUZLA_UNSAT:
              *status = triton::engines::solver::UNSAT;
              break;
            case BITWUZLA_UNKNOWN:
              *status = p.status;
              break;
          }
        }

        std::vector<std::unordered_map<triton::usize, SolverModel>> ret;
        while(res == BITWUZLA_SAT && limit >= 1) {
          std::vector<SolverTerm > solution;
          solution.reserve(_bzlaAst.getVariables().size());

          // Parse model.
          std::unordered_map<triton::usize, SolverModel> model;
          for (const auto& it : _bzlaAst.getVariables()) {
            const char* svalue = bitwuzla_get_bv_value(_solver, it.first);
            auto value = this->fromBvalueToUint512(svalue);

            auto m = SolverModel(it.second, value);
            if (model.find(m.getId()) != model.end() && model[m.getId()].getValue() != m.getValue()) {
              throw triton::exceptions::SolverEngine("Inconsistent bitwuzla model.");
            }
            model[m.getId()] = m;

            // Negate current model to escape duplication in the next solution.
            const auto& symvar_sort = _bzlaAst.getBitvectorSorts().at(it.second->getSize());
#ifdef TRITON_BOOLECTOR_INTERFACE
            auto cur_val = boolector_const(_solver, svalue);
            auto n = boolector_eq(_solver, it.first, cur_val);
            solution.push_back(boolector_not(_solver, n));
            boolector_free_bits(_solver, svalue);
#else
            auto cur_val = bitwuzla_mk_bv_value(_solver, symvar_sort, svalue, BITWUZLA_BV_BASE_BIN);
            auto n = bitwuzla_mk_term2(_solver, BITWUZLA_KIND_EQUAL, it.first, cur_val);
            solution.push_back(bitwuzla_mk_term1(_solver, BITWUZLA_KIND_NOT, n));
#endif
          }

          // Check that model is available.
          if (model.empty()) {
            break;
          }

          // Push model.
          ret.push_back(model);

          if (--limit) {
            throw triton::exceptions::SolverEngine("Multiple solutions not yet implemented for BitwuzlaCachingSolver");
            // Escape last model.
            // TODO: These have to be switched to temporary assumptions 
            if (solution.size() > 1) {
#ifdef TRITON_BOOLECTOR_INTERFACE
              // todo
              BoolectorNode *escape = NULL;
#else
              auto escape = bitwuzla_mk_term(_solver, BITWUZLA_KIND_OR, solution.size(), solution.data());
#endif
              bitwuzla_assert(_solver, escape);
            }
            else {
              bitwuzla_assert(_solver, solution.front());
            }

            // Get next model.
            res = bitwuzla_check_sat(_solver);
          }
        }

        // Get time of solving end.
        auto end = std::chrono::system_clock::now();

        if (solvingTime)
          *solvingTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        return ret;
      }

      triton::uint512 BitwuzlaCachingSolver::evaluate(const triton::ast::SharedAbstractNode& node) {
        if (node == nullptr) {
          throw triton::exceptions::AstLifting("BitwuzlaSolver::evaluate(): node cannot be null.");
        }

        // Evaluate concrete AST in solver.
        auto converted = convert(_bzlaAst, _solver, node);
        const char *bv_value;
        if (TERM_IS_VALUE(converted, _solver)) {
          bv_value = bitwuzla_get_const_bv_value(_solver, converted);
        } else {
          bv_value = bitwuzla_get_bv_value(_solver, bitwuzla_get_value(_solver, converted));
        }

#ifdef TRITON_BOOLECTOR_INTERFACE
        boolector_free_bits(_solver, bv_value);
#endif
        auto res = this->fromBvalueToUint512(bv_value);

        return res;
      }

      bool BitwuzlaCachingSolver::isConst(const triton::ast::SharedAbstractNode &node) {
        auto converted = convert(_bzlaAst, _solver, node);
        return TERM_IS_VALUE(converted, _solver);
      }

      void BitwuzlaCachingSolver::printTerm(const triton::ast::SharedAbstractNode &node) {
        auto converted = convert(_bzlaAst, _solver, node);
#ifdef TRITON_BOOLECTOR_INTERFACE
        boolector_dump_smt2_node(_solver, stdout, converted);
#else
        bitwuzla_term_dump(converted, "smt2", stdout);
#endif
        fflush(stdout);
      }

      void BitwuzlaCachingSolver::preConvert(const triton::ast::SharedAbstractNode &node) {
        convert(_bzlaAst, _solver, node);
        if (_nonincremental) {
          _nonincremental->preConvert(node);
        }
      }


      bool BitwuzlaCachingSolver::isSat(const triton::ast::SharedAbstractNode& node, triton::engines::solver::status_e* status, triton::uint32 timeout, triton::uint32* solvingTime) {
        triton::engines::solver::status_e st;

        this->getModels(node, 0, &st, timeout, solvingTime);

        if (status) {
          *status = st;
        }
        return st == triton::engines::solver::SAT;
      }

      bool BitwuzlaCachingSolver::simplify() {
        // TODO: Simplify does not seem useful...
        return true;
        // return bitwuzla_simplify(_solver) != BITWUZLA_UNSAT;
      }


      std::unordered_map<triton::usize, SolverModel> BitwuzlaCachingSolver::getModel(const triton::ast::SharedAbstractNode& node, triton::engines::solver::status_e* status, triton::uint32 timeout, triton::uint32* solvingTime) {
        auto models = this->getModels(node, 1, status, timeout, solvingTime);
        return models.empty() ? std::unordered_map<triton::usize, SolverModel>() : models.front();
      }

      void BitwuzlaCachingSolver::dumpFormula(FILE* f, const SharedAbstractNode &extra_constraint, const std::vector<SharedAbstractNode> &save_terms) {
        if (_nonincremental) {
          _nonincremental->dumpFormula(f, extra_constraint, save_terms);
          return;
        }
        bitwuzla_simplify(_solver);
        auto converted = convert(_bzlaAst, _solver, extra_constraint);

        std::vector<SolverTerm > save_terms_converted;
        for (auto &t : save_terms) {
          auto save_cvtd = convert(_bzlaAst, _solver, t);
          save_terms_converted.push_back(save_cvtd);
        }

        bitwuzla_dump_formula_and_term(_solver, converted, save_terms_converted.data(), save_terms_converted.size(), f);
      }

      void BitwuzlaCachingSolver::dumpTerm(const triton::ast::SharedAbstractNode &node, FILE* f) {
        if (_nonincremental) {
          _nonincremental->dumpTerm(node, f);
          return;
        }
        auto converted = convert(_bzlaAst, _solver, node);
#ifdef TRITON_BOOLECTOR_INTERFACE
        boolector_dump_btor_node(_solver, f, converted);
#else
        bitwuzla_term_dump(converted, "btor", f);
#endif
      }

      void BitwuzlaCachingSolver::getUnsatCore(triton::ast::SharedAstContext &ast) {
        if (!_unsat_core_enabled) {
          printf("No unsat core enabled, dumping...\n");
          FILE *formula_f = fopen("formulas/unsat-dump", "w");
          if (formula_f) {
              this->dumpFormula(formula_f, ast->equal(ast->bv(1, 1), ast->bv(1, 1)), {});
              fclose(formula_f);
          }
          exit(1);
        }
        size_t unsat_core_size;
#ifdef TRITON_BOOLECTOR_INTERFACE
        SolverTerm *unsat_core = boolector_get_failed_assumptions(_solver);
        unsat_core_size = 0;
        while (unsat_core[unsat_core_size]) {
          unsat_core_size++;
        }
#else
        SolverTerm *unsat_core = bitwuzla_get_unsat_core(_solver, &unsat_core_size);
#endif

        printf("Unsat Core: {");
        for (uint32_t i = 0; i < unsat_core_size; ++i)
        {
          printf("%d: ", i);
#ifdef TRITON_BOOLECTOR_INTERFACE
          boolector_dump_smt2_node(_solver, stdout, unsat_core[i]);
#else
          bitwuzla_term_dump(unsat_core[i], "smt2", stdout);
#endif
          printf("\n");
        }
      }

      void BitwuzlaCachingSolver::enable_unsat_core() {
        std::cout << "enable unsat core\n";
#ifdef TRITON_BOOLECTOR_INTERFACE
        throw triton::exceptions::SolverEngine("Unsat core not supported for boolector");
#else
        bitwuzla_set_option(_solver, BITWUZLA_OPT_PRODUCE_UNSAT_CORES, 1);
#endif
        _unsat_core_enabled = true;
      }
    };
  };
};
