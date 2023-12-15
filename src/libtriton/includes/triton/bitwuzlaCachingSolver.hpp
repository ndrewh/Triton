#ifndef TRITON_BITWUZLACACHINGSOLVER_H
#define TRITON_BITWUZLACACHINGSOLVER_H

#include <triton/bitwuzlaSolver.hpp>

#ifdef TRITON_BOOLECTOR_INTERFACE
#include <triton/tritonToBoolector.hpp>
#else
#include <triton/tritonToBitwuzla.hpp>
#endif

//! The Triton namespace
namespace triton {
/*!
 *  \addtogroup triton
 *  @{
 */
  //! The Engines namespace
  namespace engines {
  /*!
   *  \ingroup triton
   *  \addtogroup engines
   *  @{
   */
    //! The Solver namespace
    namespace solver {
    /*!
     *  \ingroup engines
     *  \addtogroup solver
     *  @{
     */

      using triton::ast::SharedAbstractNode;

#ifdef TRITON_BOOLECTOR_INTERFACE
      using SolverTerm = BoolectorNode*;
      using SolverPtr = Btor*;
      using TritonToAst = triton::ast::TritonToBoolector;
#else
      using SolverTerm = const BitwuzlaTerm*;
      using SolverPtr = Bitwuzla*;
      using TritonToAst = triton::ast::TritonToBitwuzla;
#endif

      
      //! \class Z3CachingSolver
      /*! \brief solver engine using z3. you can add permanent constraints using add_permanent! */
      class BitwuzlaCachingSolver : public BitwuzlaSolver {
        private:
          bool _incremental;
          bool _unsat_core_enabled;

          // HACK: Since bitwuzla doesn't support printing out formulas in incremental mode... we duplicate all add_permanent calls to this solver
          // if we are in incremental mode.
        public:
          std::unique_ptr<BitwuzlaCachingSolver> _nonincremental;
          SolverPtr _solver;
          TritonToAst _bzlaAst;

          TRITON_EXPORT BitwuzlaCachingSolver(bool incremental = true, bool needs_nonincremental_shadow = true);
          TRITON_EXPORT ~BitwuzlaCachingSolver();
          TRITON_EXPORT BitwuzlaCachingSolver(const BitwuzlaCachingSolver &) = delete; // no copy
          TRITON_EXPORT BitwuzlaCachingSolver& operator=(const BitwuzlaCachingSolver &) = delete; // no copy

          TRITON_EXPORT BitwuzlaCachingSolver(BitwuzlaCachingSolver &&); // move-construct
          TRITON_EXPORT BitwuzlaCachingSolver& operator=(BitwuzlaCachingSolver &&); // move-assign

          bool printSAT = false;
          TRITON_EXPORT void add_permanent(const triton::ast::SharedAbstractNode& node);
          TRITON_EXPORT void substitute(const triton::ast::SharedAbstractNode& varnode, const triton::ast::SharedAbstractNode &new_val);
          TRITON_EXPORT std::vector<std::unordered_map<triton::usize, SolverModel>> getModels(const triton::ast::SharedAbstractNode& node, triton::uint32 limit, triton::engines::solver::status_e* status, triton::uint32 timeout, triton::uint32* solvingTime);
          TRITON_EXPORT std::unordered_map<triton::usize, SolverModel> getModel(const triton::ast::SharedAbstractNode& node, triton::engines::solver::status_e* status = nullptr, triton::uint32 timeout = 0, triton::uint32* solvingTime = nullptr);
          TRITON_EXPORT bool isSat(const triton::ast::SharedAbstractNode& node, triton::engines::solver::status_e* status, triton::uint32 timeout, triton::uint32* solvingTime);
          //! Evaluates a Triton's AST via Bitwuzla and returns a concrete value.
          TRITON_EXPORT triton::uint512 evaluate(const triton::ast::SharedAbstractNode& node);
          TRITON_EXPORT bool isConst(const triton::ast::SharedAbstractNode& node);
          TRITON_EXPORT bool simplify();
          TRITON_EXPORT void printTerm(const triton::ast::SharedAbstractNode &node);
          TRITON_EXPORT void dumpFormula(FILE* f, const SharedAbstractNode &extra_constraint, const std::vector<SharedAbstractNode> &save_terms);
          TRITON_EXPORT void dumpTerm(const triton::ast::SharedAbstractNode &node, FILE *f);
          TRITON_EXPORT void preConvert(const triton::ast::SharedAbstractNode &node);
          TRITON_EXPORT void getUnsatCore(triton::ast::SharedAstContext &ctx);
          TRITON_EXPORT void enable_unsat_core();
      };
    }
  }
}
#endif // TRITON_BITWUZLACACHINGSOLVER_H
