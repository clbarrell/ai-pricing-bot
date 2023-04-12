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
  `You are a helpful and talkative AI expert in B2B Saas pricing and packaging approaches.
You are given the following extracted parts of podcast transcripts between pricing and packaging experts with a question. 
Provide a conversational answer. Include a short relevant quote with the timestamp for each point in your response and format as a blockquote. End your response with a suggested next question on the same topic, prefixed by "Suggested next question: ".
If you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.
If the question is not about B2B Saas pricing and packaging, politely inform them that you are tuned to only answer questions about pricing and packaging.
Question: {question}
=========
Context:
{context}
=========
Helpful Answer in Markdown:`
);

export const makeChain = (
  vectorstore: HNSWLib,
  onTokenStream?: (
    token: string,
    verbose?: boolean | undefined
  ) => Promise<void>
) => {
  const questionGenerator = new LLMChain({
    llm: new OpenAI({ temperature: 0.1, maxTokens: 2000 }),
    prompt: CONDENSE_PROMPT,
  });
  const docChain = loadQAChain(
    new OpenAI({
      temperature: 0.1,
      streaming: Boolean(onTokenStream),
      maxTokens: 2000,
    }),
    { prompt: QA_PROMPT }
  );

  return new ChatVectorDBQAChain({
    vectorstore,
    combineDocumentsChain: docChain,
    questionGeneratorChain: questionGenerator,
    returnSourceDocuments: true,
  });
};
