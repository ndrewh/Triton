//! \file
/*
**  Copyright (C) - Triton
**
**  This program is under the terms of the Apache License 2.0.
*/

#include <vector>
#include <stack>

#include <triton/coreUtils.hpp>
#include <triton/cpuSize.hpp>
#include <triton/exceptions.hpp>
#include <triton/symbolicExpression.hpp>
#include <triton/symbolicVariable.hpp>
#include <triton/tritonToBitwuzla.hpp>
#include <triton/astContext.hpp>


namespace triton {
  namespace ast {

    TritonToBitwuzla::TritonToBitwuzla(bool eval, bool caching)
      : isEval(eval), isCaching(caching) {
    }


    TritonToBitwuzla::~TritonToBitwuzla() {
      this->translatedNodes.clear();
      this->variables.clear();
      this->symbols.clear();
    }


    const std::unordered_map<const BitwuzlaTerm*, triton::engines::symbolic::SharedSymbolicVariable>& TritonToBitwuzla::getVariables(void) const {
      return this->variables;
    }


    const std::map<size_t, const BitwuzlaSort*>& TritonToBitwuzla::getBitvectorSorts(void) const {
      return this->bvSorts;
    }

    const BitwuzlaTerm* TritonToBitwuzla::convert(const SharedAbstractNode& node, Bitwuzla* bzla) {
      // if (isCaching && node->getType() != VARIABLE_NODE && translatedNodes.find(node) != translatedNodes.end()) {
      //   return this->translatedNodes.at(node);
      // }

      auto nodes = this->cachedChildTraversal(node);

      for (auto&& n : nodes) {
        if (!isCaching || n->getType() == VARIABLE_NODE || translatedNodes.find(n) == translatedNodes.end()) {
          this->translatedNodes[n] = translate(n, bzla);
        }
      }

      return this->translatedNodes.at(node);
    }

    void TritonToBitwuzla::substitute(const SharedAbstractNode& varnode, const BitwuzlaTerm *new_term, Bitwuzla* bzla) {
      if (varnode->getType() != VARIABLE_NODE) {
        // For non-variables, just replace in the translation cache
        // and if it already existed, add a constraint
        if (this->translatedNodes.find(varnode) != this->translatedNodes.end()) {
          // Set nodes equal
          auto eq = bitwuzla_mk_term2(bzla, BITWUZLA_KIND_EQUAL, this->translatedNodes[varnode], new_term);
          bitwuzla_assert(bzla, eq);
        }

        // invalidateUses(varnode);

        // Replace in cache
        this->translatedNodes[varnode] = new_term;
        return;
      }
      const auto& symVar = reinterpret_cast<VariableNode*>(varnode.get())->getSymbolicVariable();
      auto id = symVar->getId();
      auto uses = _var_uses[id];

      if (_var_cache.find(id) == _var_cache.end()) {
        convert(varnode, bzla);
      }
      auto bzla_var = _var_cache[id];
      if (_var_subs.find(id) != _var_subs.end()) {
        if (_var_subs[id] == new_term) {
            std::cout << "Sub not needed\n" << std::flush;
            return;
        }
        std::cout << "Override substitution\n" << std::flush;
        bzla_var = _var_subs[id];
      }

      auto eq = bitwuzla_mk_term2(bzla, BITWUZLA_KIND_EQUAL, bzla_var, new_term);
      bitwuzla_assert(bzla, eq);
      // _var_uses.erase(id);

      // Record the new term
      _var_subs[id] = new_term;
      // invalidateUses(varnode);
      this->translatedNodes[varnode] = new_term;
    }

    const BitwuzlaTerm* TritonToBitwuzla::translate(const SharedAbstractNode& node, Bitwuzla* bzla) {
      if (node == nullptr)
        throw triton::exceptions::AstLifting("TritonToBitwuzla::translate(): node cannot be null.");

      std::vector<const BitwuzlaTerm*> children;
      for (auto&& n : node->getChildren()) {
        auto translated = this->translatedNodes.at(n);
        children.emplace_back(translated);
        if (n->getType() == VARIABLE_NODE) {
          const auto& symVar = reinterpret_cast<VariableNode*>(n.get())->getSymbolicVariable();
          _var_uses[symVar->getId()].push_back(translated);
        }
      }

      if (node->isArray()) {
          throw triton::exceptions::AstLifting("TritonToBitwuzla::translate(): ope");
      }

      static int last_bound_var = 0;

      switch (node->getType()) {

        case ARRAY_NODE: {
          auto size  = triton::ast::getInteger<triton::uint32>(node->getChildren()[0]);
          auto isort = bitwuzla_mk_bv_sort(bzla, size);               // index sort
          auto vsort = bitwuzla_mk_bv_sort(bzla, 8);                  // value sort
          auto asort = bitwuzla_mk_array_sort(bzla, isort, vsort);    // array sort
          auto value = bitwuzla_mk_bv_value_uint64(bzla, vsort, 0);   // const value
          return bitwuzla_mk_const_array(bzla, asort, value);
        }

        case BSWAP_NODE: {
          auto bvsize = node->getBitvectorSize();
          auto* bvsort = bitwuzla_mk_bv_sort(bzla, bvsize);
          const BitwuzlaTerm *retval;

          // TODO: There's a decent chance this node itself is actually a concat... in which case we can just re-order the bits!
          retval = bitwuzla_mk_term2(bzla, BITWUZLA_KIND_BV_AND, children[0], bitwuzla_mk_bv_value_uint64(bzla, bvsort, 0xff));
          for (triton::uint32 index = 8 ; index != bvsize ; index += triton::bitsize::byte) {
            retval = bitwuzla_mk_term2(bzla, BITWUZLA_KIND_BV_SHL, retval, bitwuzla_mk_bv_value_uint64(bzla, bvsort, 8));
            retval = bitwuzla_mk_term2(bzla, BITWUZLA_KIND_BV_OR, retval,
                      bitwuzla_mk_term2(bzla, BITWUZLA_KIND_BV_AND,
                        bitwuzla_mk_term2(bzla, BITWUZLA_KIND_BV_SHR, children[0], bitwuzla_mk_bv_value_uint64(bzla, bvsort, index)),
                        bitwuzla_mk_bv_value_uint64(bzla, bvsort, 0xff)
                      )
                     );
          }

          return retval;
        }

        case BVADD_NODE:
          return bitwuzla_mk_term2(bzla, BITWUZLA_KIND_BV_ADD, children[0], children[1]);

        case BVAND_NODE:
          return bitwuzla_mk_term2(bzla, BITWUZLA_KIND_BV_AND, children[0], children[1]);

        case BVASHR_NODE:
          return bitwuzla_mk_term2(bzla, BITWUZLA_KIND_BV_ASHR, children[0], children[1]);

        case BVLSHR_NODE:
          return bitwuzla_mk_term2(bzla, BITWUZLA_KIND_BV_SHR, children[0], children[1]);

        case BVMUL_NODE:
          return bitwuzla_mk_term2(bzla, BITWUZLA_KIND_BV_MUL, children[0], children[1]);

        case BVNAND_NODE:
          return bitwuzla_mk_term2(bzla, BITWUZLA_KIND_BV_NAND, children[0], children[1]);

        case BVNEG_NODE:
          return bitwuzla_mk_term1(bzla, BITWUZLA_KIND_BV_NEG, children[0]);

        case BVNOR_NODE:
          return bitwuzla_mk_term2(bzla, BITWUZLA_KIND_BV_NOR, children[0], children[1]);

        case BVNOT_NODE:
          return bitwuzla_mk_term1(bzla, BITWUZLA_KIND_BV_NOT, children[0]);

        case BVOR_NODE:
          return bitwuzla_mk_term2(bzla, BITWUZLA_KIND_BV_OR, children[0], children[1]);

        case BVROL_NODE: {
          auto childNodes = node->getChildren();
          auto idx = triton::ast::getInteger<triton::usize>(childNodes[1]);
          return bitwuzla_mk_term1_indexed1(bzla, BITWUZLA_KIND_BV_ROLI, children[0], idx);
        }

        case BVROR_NODE: {
          auto childNodes = node->getChildren();
          auto idx = triton::ast::getInteger<triton::usize>(childNodes[1]);
          return bitwuzla_mk_term1_indexed1(bzla, BITWUZLA_KIND_BV_RORI, children[0], idx);
        }

        case BVSDIV_NODE:
          return bitwuzla_mk_term2(bzla, BITWUZLA_KIND_BV_SDIV, children[0], children[1]);

        case BVSGE_NODE:
          return bitwuzla_mk_term2(bzla, BITWUZLA_KIND_BV_SGE, children[0], children[1]);

        case BVSGT_NODE:
          return bitwuzla_mk_term2(bzla, BITWUZLA_KIND_BV_SGT, children[0], children[1]);

        case BVSHL_NODE: {
          auto shiftSize = node->getChildren()[1];

          // ahaberlandt: Bitwuzla struggles to reason about arithmetic over BSHL. Convert these to concat over extracts, which
          // bitwuzla does in fact reason about arithmetic over.
          if (!shiftSize->isSymbolized()) {
            auto shiftSizeI = (uint64_t)shiftSize->evaluate();
            if (shiftSizeI && shiftSizeI % 8 == 0 && shiftSizeI < node->getBitvectorSize()) {
              // Turn it into a concat + extract
              auto extract = bitwuzla_mk_term1_indexed2(bzla, BITWUZLA_KIND_BV_EXTRACT, children[0], node->getBitvectorSize() - shiftSizeI - 1, 0);
              auto zero_size = shiftSizeI;
              auto zero_sort = this->bvSorts.find(zero_size);
              if (zero_sort == this->bvSorts.end()) {
                zero_sort = this->bvSorts.insert({zero_size, bitwuzla_mk_bv_sort(bzla, zero_size)}).first;
              }
              auto zero = bitwuzla_mk_bv_zero(bzla, zero_sort->second);
              const BitwuzlaTerm *concat_nodes[2] = { extract, zero };
              auto concat = bitwuzla_mk_term(bzla, BITWUZLA_KIND_BV_CONCAT, 2, concat_nodes);
              return concat;
            }
          }
          return bitwuzla_mk_term2(bzla, BITWUZLA_KIND_BV_SHL, children[0], children[1]);
        }

        case BVSLE_NODE:
          return bitwuzla_mk_term2(bzla, BITWUZLA_KIND_BV_SLE, children[0], children[1]);

        case BVSLT_NODE:
          return bitwuzla_mk_term2(bzla, BITWUZLA_KIND_BV_SLT, children[0], children[1]);

        case BVSMOD_NODE:
          return bitwuzla_mk_term2(bzla, BITWUZLA_KIND_BV_SMOD, children[0], children[1]);

        case BVSREM_NODE:
          return bitwuzla_mk_term2(bzla, BITWUZLA_KIND_BV_SREM, children[0], children[1]);

        case BVSUB_NODE:
          return bitwuzla_mk_term2(bzla, BITWUZLA_KIND_BV_SUB, children[0], children[1]);

        case BVUDIV_NODE:
          return bitwuzla_mk_term2(bzla, BITWUZLA_KIND_BV_UDIV, children[0], children[1]);

        case BVUGE_NODE:
          return bitwuzla_mk_term2(bzla, BITWUZLA_KIND_BV_UGE, children[0], children[1]);

        case BVUGT_NODE:
          return bitwuzla_mk_term2(bzla, BITWUZLA_KIND_BV_UGT, children[0], children[1]);

        case BVULE_NODE:
          return bitwuzla_mk_term2(bzla, BITWUZLA_KIND_BV_ULE, children[0], children[1]);

        case BVULT_NODE:
          return bitwuzla_mk_term2(bzla, BITWUZLA_KIND_BV_ULT, children[0], children[1]);

        case BVUREM_NODE:
          return bitwuzla_mk_term2(bzla, BITWUZLA_KIND_BV_UREM, children[0], children[1]);

        case BVXNOR_NODE:
          return bitwuzla_mk_term2(bzla, BITWUZLA_KIND_BV_XNOR, children[0], children[1]);

        case BVXOR_NODE:
          return bitwuzla_mk_term2(bzla, BITWUZLA_KIND_BV_XOR, children[0], children[1]);

        case BV_NODE: {
          auto childNodes = node->getChildren();
          auto bv_size = triton::ast::getInteger<triton::usize>(childNodes[1]);
          auto sort = this->bvSorts.find(bv_size);
          if (sort == this->bvSorts.end()) {
            sort = this->bvSorts.insert({bv_size, bitwuzla_mk_bv_sort(bzla, bv_size)}).first;
          }

          // Handle bitvector value as integer if it small enough.
          if (bv_size <= sizeof(uint64_t) * 8) {
            auto bv_value = triton::ast::getInteger<triton::uint64>(childNodes[0]);
            return bitwuzla_mk_bv_value_uint64(bzla, sort->second, bv_value);
          }

          auto bv_value = triton::ast::getInteger<std::string>(childNodes[0]);
          return bitwuzla_mk_bv_value(bzla, sort->second, bv_value.c_str(), BITWUZLA_BV_BASE_DEC);
        }

        case CONCAT_NODE:
          return bitwuzla_mk_term(bzla, BITWUZLA_KIND_BV_CONCAT, children.size(), children.data());

        case DISTINCT_NODE:
          return bitwuzla_mk_term2(bzla, BITWUZLA_KIND_DISTINCT, children[0], children[1]);

        case EQUAL_NODE:
          return bitwuzla_mk_term2(bzla, BITWUZLA_KIND_EQUAL, children[0], children[1]);

        case EXTRACT_NODE: {
          auto childNodes = node->getChildren();
          auto high = triton::ast::getInteger<triton::usize>(childNodes[0]);
          auto low  = triton::ast::getInteger<triton::usize>(childNodes[1]);
          return bitwuzla_mk_term1_indexed2(bzla, BITWUZLA_KIND_BV_EXTRACT, children[2], high, low);
        }

        case FORALL_NODE:
          throw triton::exceptions::AstLifting("TritonToBitwuzla::translate(): FORALL node can't be converted due to a Bitwuzla issue (see #1062).");

        case IFF_NODE:
          return bitwuzla_mk_term2(bzla, BITWUZLA_KIND_IFF, children[0], children[1]);

        case INTEGER_NODE:
          return nullptr;

        case ITE_NODE:
          return bitwuzla_mk_term3(bzla, BITWUZLA_KIND_ITE, children[0], children[1], children[2]);

        case LAND_NODE:
          return bitwuzla_mk_term(bzla, BITWUZLA_KIND_AND, children.size(), children.data());

        case LET_NODE: {
          auto childNodes = node->getChildren();
          symbols[reinterpret_cast<triton::ast::StringNode*>(childNodes[0].get())->getString()] = childNodes[1];
          return children[2];
        }

        case LNOT_NODE:
          return bitwuzla_mk_term1(bzla, BITWUZLA_KIND_NOT, children[0]);

        case LOR_NODE:
          return bitwuzla_mk_term(bzla, BITWUZLA_KIND_OR, children.size(), children.data());

        case LXOR_NODE:
          return bitwuzla_mk_term(bzla, BITWUZLA_KIND_XOR, children.size(), children.data());

        case REFERENCE_NODE: {
          auto ref = reinterpret_cast<ReferenceNode*>(node.get())->getSymbolicExpression()->getAst();
          return this->translatedNodes.at(ref);
        }

        case SELECT_NODE:
          return bitwuzla_mk_term2(bzla, BITWUZLA_KIND_ARRAY_SELECT, children[0], children[1]);

        case STORE_NODE:
          return bitwuzla_mk_term3(bzla, BITWUZLA_KIND_ARRAY_STORE, children[0], children[1], children[2]);

        case STRING_NODE: {
          std::string value = reinterpret_cast<triton::ast::StringNode*>(node.get())->getString();

          auto it = symbols.find(value);
          if (it == symbols.end())
            throw triton::exceptions::AstLifting("TritonToBitwuzla::translate(): [STRING_NODE] Symbols not found.");

          return this->translatedNodes.at(it->second);
        }

        case SX_NODE: {
          auto childNodes = node->getChildren();
          auto ext = triton::ast::getInteger<triton::usize>(childNodes[0]);
          return bitwuzla_mk_term1_indexed1(bzla, BITWUZLA_KIND_BV_SIGN_EXTEND, children[1], ext);
        }

        case VARIABLE_NODE: {
          const auto& symVar = reinterpret_cast<VariableNode*>(node.get())->getSymbolicVariable();

          if (_var_subs.find(symVar->getId()) != _var_subs.end()) {
            return _var_subs[symVar->getId()];
          }
          auto size = symVar->getSize();
          auto sort = this->bvSorts.find(size);
          if (sort == this->bvSorts.end()) {
            sort = this->bvSorts.insert({size, bitwuzla_mk_bv_sort(bzla, size)}).first;
          }

          // If the conversion is used to evaluate a node, we concretize symbolic variables.
          if (this->isEval) {
            triton::uint512 value = reinterpret_cast<triton::ast::VariableNode*>(node.get())->evaluate();
            if (size <= sizeof(uint64_t) * 8) {
              return bitwuzla_mk_bv_value_uint64(bzla, sort->second, static_cast<uint64_t>(value));
            }
            return bitwuzla_mk_bv_value(bzla, sort->second, triton::utils::toString(value).c_str(), BITWUZLA_BV_BASE_DEC);
          }

          // OK so here's the deal: if you call bitwuzla_mk_const twice with the same
          // name you appear to get different variables. So we have to make a cache
          // so that this doesn't happen.
          //
          // However, if you make an incremental solver (using bitwuzla_assume) then
          // these variables, if not referenced elsewhere, will get free'd
          auto it = _var_cache.find(symVar->getId());
          if (it != _var_cache.end()) {
            return it->second;
          }

          auto n = bitwuzla_mk_const(bzla, sort->second, symVar->getName().c_str());
          variables[n] = symVar;
          _var_cache[symVar->getId()] = n;

          // We have to do this because it could be pulled from the _var_cache
          // after being released as an bzla_assumption

          // TODO/FIXME: We have to decrement refcount in destructor
          // bitwuzla_increment_refcount(bzla, n);
          return n;
        }

        case ZX_NODE: {
          auto childNodes = node->getChildren();
          auto ext = triton::ast::getInteger<triton::usize>(childNodes[0]);
          return bitwuzla_mk_term1_indexed1(bzla, BITWUZLA_KIND_BV_ZERO_EXTEND,children[1], ext);
        }

        default:
          throw triton::exceptions::AstLifting("TritonToBitwuzla::translate(): Invalid kind of node.");
      }
    }

    std::vector<SharedAbstractNode> TritonToBitwuzla::cachedChildTraversal(const SharedAbstractNode& node) {
      std::vector<SharedAbstractNode> result;
      std::unordered_set<AbstractNode*> visited;
      std::stack<std::pair<SharedAbstractNode, bool>> worklist;
      const bool revert = true;
      const bool descend = true;
      const bool unroll = true;

      // if (this->translatedNodes.find(node) != this->translatedNodes.end()) {
      //   // Translation is available for the root. Do not descend.
      //   return {};
      // }

      if (node == nullptr)
        throw triton::exceptions::Ast("triton::ast::nodesExtraction(): Node cannot be null.");

      /*
       *  We use a worklist strategy to avoid recursive calls
       *  and so stack overflow when going through a big AST.
       */
      worklist.push({node, false});

      while (!worklist.empty()) {
        SharedAbstractNode ast;
        bool postOrder;
        std::tie(ast, postOrder) = worklist.top();
        worklist.pop();

        /* It means that we visited all children of this node and we can put it in the result */
        if (postOrder) {
          result.push_back(ast);
          continue;
        }

        if (!visited.insert(ast.get()).second) {
          continue;
        }

        worklist.push({ast, true});

        const auto& relatives = descend ? ast->getChildren() : ast->getParents();

        /* Proceed relatives */
        for (const auto& r : relatives) {
          if (visited.find(r.get()) == visited.end() && this->translatedNodes.find(r) == this->translatedNodes.end()) {
            worklist.push({r, false});
          }
        }

        /* If unroll is true, we unroll all references */
        if (unroll && ast->getType() == REFERENCE_NODE) {
          const SharedAbstractNode& ref = reinterpret_cast<ReferenceNode*>(ast.get())->getSymbolicExpression()->getAst();
          if (visited.find(ref.get()) == visited.end() && this->translatedNodes.find(ref) == this->translatedNodes.end()) {
            worklist.push({ref, false});
          }
        }
      }

      /* The result is in reversed topological sort meaning that children go before parents */
      if (!revert) {
        std::reverse(result.begin(), result.end());
      }

      return result;
    }

    void TritonToBitwuzla::invalidateUses(const SharedAbstractNode &var) {
      auto all_parents = triton::ast::parentsExtraction(var, true);
      int use_count = 0;
      for (auto &ast : all_parents) {
        if (ast->isArray()) {
          throw triton::exceptions::AstLifting("TritonToBitwuzla::invalidateUses(): OPE");
        }
        if (ast != var) {
          this->translatedNodes.erase(ast);
          use_count++;
        }
      }
      std::cout << "invalidateUses " << use_count << "\n";
    }

  }; /* ast namespace */
}; /* triton namespace */
