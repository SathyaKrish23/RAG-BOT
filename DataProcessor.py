from PdfReader import read_pdf
from Chunker import chunk_pages
from Embedder import embed_chunks
from VectorStore import store_in_pinecone


pdf_path = "./resources/realistic_hr_policy.pdf"
def run():
    # Read HR Policy PDF and extract text
    pages = read_pdf(pdf_path)

    # Chunk the extracted text into manageable pieces
    chunks = chunk_pages(pages, chunk_size=900, chunk_overlap=150)

    # embed the chunks 
    embedded_chunks = embed_chunks(chunks)


    # store the embedded chunks
    store_in_pinecone(chunks, embedded_chunks, namespace="")
   
    
if __name__ == "__main__":
    run()