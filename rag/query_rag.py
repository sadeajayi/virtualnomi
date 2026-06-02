#!/usr/bin/env python3
"""
Interactive query interface for the RAG system.
Query Yoruba research papers for cultural context, morpheme analysis, and semantic information.
"""

from rag_service import YorubaRAGService
import sys

def print_results(results, max_results=3):
    """Pretty print search results"""
    if not results:
        print("   ⚠️  No results found")
        return
    
    print(f"   ✅ Found {len(results)} results:\n")
    for i, result in enumerate(results[:max_results], 1):
        print(f"   {i}. From: {result['paper']}")
        if 'similarity' in result:
            print(f"      Relevance: {result['similarity']:.3f}")
        
        # Clean up text for display
        text = result['text'].replace('\n', ' ').strip()
        if len(text) > 200:
            text = text[:200] + "..."
        print(f"      {text}\n")

def query_rag_interactive():
    """Interactive query interface"""
    
    print("=" * 80)
    print("🔍 YORUBA RAG QUERY INTERFACE")
    print("=" * 80)
    print()
    
    # Initialize RAG
    try:
        rag = YorubaRAGService()
        print("✅ RAG service initialized\n")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}\n")
        return
    
    print("Available commands:")
    print("  • search <query>     - Semantic search")
    print("  • context <name> <meaning> - Get cultural context for a name")
    print("  • morpheme <name>    - Extract morphemes from a name")
    print("  • morpheme-search <morpheme> - Search for morpheme information")
    print("  • help               - Show this help")
    print("  • quit/exit          - Exit")
    print()
    print("Examples:")
    print("  search Yoruba naming traditions")
    print("  context Abiola 'Born into wealth'")
    print("  morpheme Adeyemi")
    print("  morpheme-search ọlá")
    print()
    
    while True:
        try:
            user_input = input("🔍 RAG> ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if user_input.lower() == 'help':
                print("\nAvailable commands:")
                print("  • search <query>     - Semantic search")
                print("  • context <name> <meaning> - Get cultural context")
                print("  • morpheme <name>    - Extract morphemes")
                print("  • morpheme-search <morpheme> - Search morpheme info")
                print("  • quit/exit          - Exit")
                print()
                continue
            
            parts = user_input.split(None, 1)
            command = parts[0].lower()
            
            if command == 'search':
                if len(parts) < 2:
                    print("   ⚠️  Usage: search <query>")
                    continue
                
                query = parts[1]
                print(f"\n🔍 Searching for: '{query}'")
                print("-" * 80)
                results = rag.search(query, top_k=5)
                print_results(results)
            
            elif command == 'context':
                if len(parts) < 2:
                    print("   ⚠️  Usage: context <name> <meaning>")
                    continue
                
                args = parts[1].split(None, 1)
                if len(args) < 2:
                    print("   ⚠️  Usage: context <name> <meaning>")
                    continue
                
                name = args[0]
                meaning = args[1].strip("'\"")
                
                print(f"\n📚 Cultural context for: {name}")
                print(f"   Meaning: \"{meaning}\"")
                print("-" * 80)
                
                context = rag.get_cultural_context(name, meaning)
                if context:
                    print(f"\n{context}\n")
                else:
                    print("   ⚠️  No cultural context found\n")
            
            elif command == 'morpheme':
                if len(parts) < 2:
                    print("   ⚠️  Usage: morpheme <name>")
                    continue
                
                name = parts[1]
                print(f"\n🔤 Morphemes in '{name}':")
                print("-" * 80)
                
                morphemes = rag._extract_morphemes(name)
                if morphemes:
                    print(f"   ✅ Found: {morphemes}\n")
                else:
                    print("   ⚠️  No morphemes found\n")
            
            elif command == 'morpheme-search':
                if len(parts) < 2:
                    print("   ⚠️  Usage: morpheme-search <morpheme>")
                    continue
                
                morpheme = parts[1]
                query = f"{morpheme} morpheme semantic meaning cultural significance"
                
                print(f"\n🔍 Searching for morpheme: '{morpheme}'")
                print("-" * 80)
                results = rag.search(query, top_k=5)
                print_results(results)
            
            else:
                print(f"   ⚠️  Unknown command: {command}")
                print("   Type 'help' for available commands")
                print()
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"   ❌ Error: {e}\n")

def query_rag_programmatic():
    """Show programmatic usage examples"""
    
    print("=" * 80)
    print("📖 PROGRAMMATIC USAGE EXAMPLES")
    print("=" * 80)
    print()
    
    code_examples = """
# 1. Initialize RAG service
from rag_service import YorubaRAGService

rag = YorubaRAGService()

# 2. Semantic search
results = rag.search("Yoruba naming traditions", top_k=5)
for result in results:
    print(f"Paper: {result['paper']}")
    print(f"Text: {result['text'][:200]}...")
    print(f"Relevance: {result.get('similarity', 0)}")

# 3. Get cultural context for a name
context = rag.get_cultural_context("Abiola", "Born into wealth")
print(context)

# 4. Extract morphemes from a name
morphemes = rag._extract_morphemes("Adeyemi")
print(morphemes)  # ['ade']

# 5. Search for morpheme-specific information
results = rag.search("ọlá morpheme semantic meaning", top_k=3)

# 6. Get relevant excerpts
excerpts = rag.get_relevant_excerpts("birth circumstances names", max_excerpts=3)
for excerpt in excerpts:
    print(f"From {excerpt['paper']}: {excerpt['excerpt'][:200]}...")
"""
    
    print(code_examples)
    print("=" * 80)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--examples':
        query_rag_programmatic()
    else:
        query_rag_interactive()


