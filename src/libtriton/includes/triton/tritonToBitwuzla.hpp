//! \file
/*
**  Copyright (C) - Triton
**
**  This program is under the terms of the Apache License 2.0.
*/

#ifndef TRITON_TRITONTOBITWUZLA_H
#define TRITON_TRITONTOBITWUZLA_H

#include <map>
#include <unordered_map>
#include <memory>

extern "C" {
#include <bitwuzla/bitwuzla.h>
}

#include <triton/ast.hpp>
#include <triton/dllexport.hpp>
#include <triton/tritonTypes.hpp>



//! The Triton namespace
namespace triton {
/*!
 *  \addtogroup triton
 *  @{
 */

  //! The AST namespace
  namespace ast {
  /*!
   *  \ingroup triton
   *  \addtogroup ast
   *  @{
   */

    //! \class TritonToBitwuzla
    /*! \brief Converts a Triton's AST to Bitwuzla's AST. */
    class TritonToBitwuzla {
      public:
        //! Constructor.
        TRITON_EXPORT TritonToBitwuzla(bool eval=false, bool caching=false);

        // delete copy-constructors
        TRITON_EXPORT TritonToBitwuzla(TritonToBitwuzla &other) = delete;
        TRITON_EXPORT TritonToBitwuzla& operator=(TritonToBitwuzla &other) = delete;

        // move-assignment
        TRITON_EXPORT TritonToBitwuzla(TritonToBitwuzla &&other) = default;
        TRITON_EXPORT TritonToBitwuzla& operator=(TritonToBitwuzla &&other) = default;

        //! Destructor.
        TRITON_EXPORT ~TritonToBitwuzla();

        //! Converts to Bitwuzla's AST
        TRITON_EXPORT const BitwuzlaTerm* convert(const SharedAbstractNode& node, Bitwuzla* bzla);

        //! Returns symbolic variables and its assosiated Bitwuzla terms to process the solver model.
        TRITON_EXPORT const std::unordered_map<const BitwuzlaTerm*, triton::engines::symbolic::SharedSymbolicVariable>& getVariables(void) const;

        //! Returns bitvector sorts.
        TRITON_EXPORT const std::map<size_t, const BitwuzlaSort*>& getBitvectorSorts(void) const;

        //! Substitute a term for a variable
        TRITON_EXPORT void substitute(const SharedAbstractNode& variable, const BitwuzlaTerm *new_term, Bitwuzla* bzla);

      private:
        //! The map of Triton's AST nodes translated to the Bitwuzla terms.
        std::map<WeakAbstractNode, const BitwuzlaTerm*, std::owner_less<WeakAbstractNode>> translatedNodes;

        //! The set of symbolic variables contained in the expression.
        std::unordered_map<const BitwuzlaTerm*, triton::engines::symbolic::SharedSymbolicVariable> variables;

        //! The set of symbolic variables contained in the expression.
        std::unordered_map<triton::usize, std::vector<const BitwuzlaTerm*>> _var_uses;

        //! The set of symbolic variables contained in the expression.
        std::unordered_map<triton::usize, const BitwuzlaTerm*> _var_subs;

        //! The map from symbolic variable id to BitwuzlaTerm (variable identity is not based on name)
        std::unordered_map<triton::usize, const BitwuzlaTerm*> _var_cache;

        //! The map of symbols. E.g: (let (symbols expr1) expr2)
        std::unordered_map<std::string, triton::ast::SharedAbstractNode> symbols;

        //! All bitvector sorts that used in the expression.
        std::map<size_t, const BitwuzlaSort*> bvSorts;

        //! This flag define if the conversion is used to evaluated a node or not.
        bool isEval;
        bool isCaching;

        //! The convert internal process.
        const BitwuzlaTerm* translate(const SharedAbstractNode& node, Bitwuzla* bzla);
        std::vector<SharedAbstractNode> cachedChildTraversal(const SharedAbstractNode& node);
        void invalidateUses(const SharedAbstractNode &var);
    };

  /*! @} End of ast namespace */
  };
/*! @} End of triton namespace */
};

#endif /* TRITON_TRITONTOBITWUZLA_H */
