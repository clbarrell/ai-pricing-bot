import { OpenAI } from "langchain/llms";
import { LLMChain, ChatVectorDBQAChain, loadQAChain } from "langchain/chains";
import { HNSWLib } from "langchain/vectorstores";
import { PromptTemplate } from "langchain/prompts";

const CONDENSE_PROMPT =
  PromptTemplate.fromTemplate(`Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:`);

const QA_PROMPT = PromptTemplate.fromTemplate(
  `You are an AI expert in B2B Saas pricing and packaging approaches.
You are given the following extracted parts of podcasts transcripts with pricing and packaging experts and a question. Provide a conversational answer. Show the context metadata when possible.
Do NOT make up a hyperlink that is not listed.
If you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.
If the question is not about B2B Saas pricing and packaging, politely inform them that you are tuned to only answer questions about pricing and packaging.
Question: {question}
=========
{context}
=========
Answer in Markdown:`
);

export const makeChain = (
  vectorstore: HNSWLib,
  onTokenStream?: (
    token: string,
    verbose?: boolean | undefined
  ) => Promise<void>
) => {
  const questionGenerator = new LLMChain({
    llm: new OpenAI({ temperature: 0.2 }),
    prompt: CONDENSE_PROMPT,
  });
  const docChain = loadQAChain(
    new OpenAI({
      temperature: 0,
      streaming: Boolean(onTokenStream),
    }),
    { prompt: QA_PROMPT }
  );

  return new ChatVectorDBQAChain({
    vectorstore,
    combineDocumentsChain: docChain,
    questionGeneratorChain: questionGenerator,
  });
};
