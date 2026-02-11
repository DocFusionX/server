HYDE_PROMPT = (
    "You are an expert Q&A system. Your task is to generate a hypothetical answer to the following question. "
    "The answer should be a short, concise paragraph that contains the kind of information you would expect to find in a document that answers the question."
)

MAP_PROMPT = (
    "You are an expert Q&A system. Your task is to answer the following question based *only* on the provided context.\n"
    "If the context does not contain the answer, state that the answer is not available in the provided context.\n"
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
