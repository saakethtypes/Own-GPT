import streamlit as st
import pinecone
import openai

st.title("GPT Based RAG KG ")

limit = 3750
openai.api_key = st.secrets["OPENAI_API_KEY"]
embed_model = "text-embedding-ada-002"


def retrieve(query):
    res = openai.Embedding.create(
        input=[query],
        engine=embed_model
    )

    # retrieve from Pinecone
    xq = res['data'][0]['embedding']

    # get relevant contexts
    res = index.query(xq, top_k=3, include_metadata=True)
    contexts = [
        x['metadata']['text'] for x in res['matches']
    ]

    # build our prompt with the retrieved contexts included
    prompt_start = (
        "Answer the question based on the context below.\n\n"+
        "Context:\n"
    )
    prompt_end = (
        f"\n\nQuestion: {query}\nAnswer:"
    )
    prompt = prompt_start + contexts[0] + prompt_end
    return prompt

def complete(prompt):
    res = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        temperature=0,
        max_tokens=400,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    return res['choices'][0]['text'].strip()

index_name = 'openaiyoutube'

# initialize connection to pinecone (get API key at app.pinecone.io)
pinecone.init(
    api_key=st.secrets["PINECONE_API_KEY"], #
    environment="us-west4-gcp"
)

# check if index already exists (it shouldn't if this is first time)
if index_name not in pinecone.list_indexes():
    # if does not exist, create index
    pinecone.create_index(
        index_name,
        dimension=len(res['data'][0]['embedding']),
        metric='cosine',
        metadata_config={'indexed': ['channel_id', 'published']}
    )

# connect to index
index = pinecone.Index(index_name)
def ask_me(prompt):
    query = prompt
    query_with_contexts = retrieve(query)
    print(query_with_contexts)
    return complete(query_with_contexts)

st.image("images.jpeg")

# user input
user_prompt = st.text_input("Enter a question related to Canon R5: ")
if len(user_prompt)>5:
    ans = ask_me(user_prompt)
    st.write(ans)
