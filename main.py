import streamlit as st
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_groq import ChatGroq
import os
groq_key = os.getenv("GROQ")

#LLM and key loading function
def load_LLM():
    # Make sure your openai_api_key is set as an environment variable
    llm = ChatGroq(
    model="qwen-2.5-32b",
    groq_api_key=groq_key)
    return llm


# Define example inputs and outputs
examples = [
    {
        "review": "This dress is pretty amazing. It arrived in two days, just in time for my wife's anniversary present. It is cheaper than the other dresses out there, but I think it is worth it for the extra features.",
        "output": "- Sentiment: Positive\n- How long took it to deliver? 2 days\n- How was the price perceived? Cheap"
    },
    {
        "review": "I'm really disappointed with this phone case. It took over a week to arrive, and for the price, I expected better quality. Definitely not worth the money.",
        "output": "- Sentiment: Negative\n- How long took it to deliver? More than a week\n- How was the price perceived? Expensive"
    },
    {
        "review": "The headphones are okay. They sound decent, and the price was reasonable. Delivery was quick, but nothing extraordinary.",
        "output": "- Sentiment: Neutral\n- How long took it to deliver? No information about this\n- How was the price perceived? Neutral"
    }
]

# Define the example template
example_template = """\
text: {review}

{output}
"""

# Create the few-shot prompt template
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=PromptTemplate(
        input_variables=["review", "output"],
        template=example_template
    ),
    prefix="For the following text, extract the following information:\n"
           "- Sentiment: Is the customer happy with the product? Answer Positive, Negative, Neutral, or Unknown.\n"
           "- How long took it to deliver? Extract delivery time if mentioned, else output 'No information about this.'\n"
           "- How was the price perceived? Answer Expensive, Cheap, Neutral, or Unknown.\n\n"
           "Examples:\n",
    suffix="text: {review}\n",
    input_variables=["review"]
)

#Page title and header
st.set_page_config(page_title="Extract Key Information from Product Reviews")
st.header("Extract Key Information from Product Reviews")

st.markdown("Extract key information from a product review.")
st.markdown("""
    - Sentiment
    - How long took it to deliver?
    - How was its price perceived?
    """)


# Input
st.markdown("## Enter the product review")

def get_review():
    review_text = st.text_area(label="Product Review", label_visibility='collapsed', placeholder="Your Product Review...", key="review_input")
    return review_text

review_input = get_review()

if len(review_input.split(" ")) > 700:
    st.write("Please enter a shorter product review. The maximum length is 700 words.")
    st.stop()

    
# Output
st.markdown("### Key Data Extracted:")

if review_input:

    llm = load_LLM()
    # Format the final prompt with the new review
    prompt_with_review = few_shot_prompt.format(review=review_input)

    key_data_extraction = llm.invoke(prompt_with_review).content

    st.write(key_data_extraction)