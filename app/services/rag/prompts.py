HYDE_PROMPT = (
    "You are an expert Q&A system. Your task is to generate a hypothetical answer to the following question. "
    "The answer should be a short, concise paragraph that contains the kind of information you would expect to find in a document that answers the question."
)

MAP_PROMPT = (
    "You are an expert Q&A system. Your task is to answer the following question based *only* on the provided context.\n"
    "If the context does not contain the information needed to answer the question, respond with an empty string.\n"
    "Question: {question}\n\n"
    "Context:\n{context}"
)

REDUCE_PROMPT = (
    "You are an expert Q&A system. Your task is to synthesize a final, comprehensive answer from a set of provided answers to a single question.\n"
    "The original question was: {question}\n\n"
    "Here are the answers generated from different pieces of context:\n"
    "{answers}\n\n"
    "Please synthesize these into a single, coherent, and comprehensive answer."
)

STUFF_PROMPT = (
    "You are an expert Q&A system. Your task is to answer the question based *only* on the provided context below.\n"
    "The context consists of document segments and may include structural information (Table of Contents or Headers).\n"
    "Pay attention to the document structure to understand the context and hierarchy of the information.\n"
    "If the context does not contain the information needed to answer the question, state that you do not know.\n\n"
    "Context:\n"
    "{context}\n\n"
    "Question: {question}"
)
