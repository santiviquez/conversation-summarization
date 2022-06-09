import streamlit as st
from transformers import pipeline


# Streamlit app title
st.markdown("## Conversation Summarization")
st.markdown("")
st.markdown(
    "Tired of having to read looooooong whatsapp messagges? **Me too!** That's way I told my computer to summarize them for me."
)

st.markdown("Go on and summarize yours!")


@st.cache(allow_output_mutation=True, hash_funcs=None)
def load_model(hub_model_id):
    summarizer = pipeline("summarization", model=hub_model_id)
    return summarizer


def summarize(text):
    summary = summarizer(text)[0]["summary_text"]
    return summary


hub_model_id = "santiviquez/bart-base-finetuned-samsum-en"

example_1 = """
Janice: my son has been asking me to get him a hamster for his birthday
Janice: should i?
Martina: NO! NO! NO! NO! NO!
Martina: i got one for my son and it stank up the whole house
Martina: so don't do it!!!'
"""

example_2 = """
Ivan: hey eric
Eric: yeah man
Ivan: so youre coming to the wedding
Eric: your brother's
Ivan: yea
Eric: i dont know mannn
Ivan: YOU DONT KNOW??
Eric: i just have a lot to do at home, plus i dont know if my parents would let me
Ivan: ill take care of your parents
Eric: youre telling me you have the guts to talk to them XD
Ivan: thats my problem
Eric: okay man, if you say so
Ivan: yea just be there 
Eric: alright'
"""

conversation = st.selectbox("Examples", ("Example 1", "Example 2", "Write your own"))

if conversation == "Example 1":
    input_text = example_1
elif conversation == "Example 2":
    input_text = example_2
else:
    input_text = ""

text = st.text_area(label="Conversation", height=200, max_chars=200, value=input_text)


if st.button("Summarize"):
    summarizer = load_model(hub_model_id)
    generated_summary = summarize(text)
    st.write("Generated Summary:")
    st.write(generated_summary)


st.text("")
st.text("")
st.text("")
st.text("Trained using HuggingFace 🤗")
st.markdown(
    "`Create by` [santiviquez](https://twitter.com/santiviquez) |  `Code:` [GitHub](https://github.com/santiviquez/conversation-summarization)"
)
