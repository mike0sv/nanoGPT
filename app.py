from random import randint
from urllib.parse import urlencode

import streamlit
import streamlit.components.v1
from pydantic import BaseModel

from mlem.runtime.client import HTTPClient
from mlem.runtime.interface import ExecutionError

MAX_CACHE = 10


@streamlit.cache(hash_funcs={HTTPClient: lambda x: 0})
def get_client():
    return HTTPClient(
        host="{{server_host}}", port=int("{{server_port}}"), raw=True
    )


client = get_client()
streamlit.set_page_config(
    page_title="nanoGPT+MLEM",
)
streamlit.title("nanoGPT MLEM Docs Generator")
streamlit.markdown("Read more in this [blogpost](todo-link)")


def get_session_id():
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    ctx = get_script_run_ctx()
    if ctx is None:
        raise Exception("Failed to get the thread context")
    return ctx.session_id


@streamlit.experimental_singleton
def _dialogue(user_id):
    return []


dialogue = _dialogue(get_session_id())

def promt():
    with streamlit.form(key="_"):
        chat, settings = streamlit.tabs(["Chat", "Settings"])
        with settings:
            max_new_tokens = streamlit.number_input("Size", min_value=1,
                                                    max_value=1000, value=50,
                                                    step=1)
            temperature = streamlit.number_input("Temperature", value=0.3)
            random_seed = streamlit.checkbox("Randomize seed", value=True)
            seed = streamlit.number_input("seed",
                                          value=1 if not random_seed else randint(
                                              0, 10000))
        with chat:
            for d in dialogue:
                streamlit.markdown(d, unsafe_allow_html=True)
            start = streamlit.text_input(label="Promt") or " "
            arg_values = {"start": start,
                          "max_new_tokens": max_new_tokens,
                          "temperature": temperature, "top_k": 100,
                          "num_samples": 1, "seed": seed}
            submit_button = streamlit.form_submit_button(label="OK")

    if submit_button:
        with streamlit.spinner("Processing..."):
            try:
                dialogue.append(f'<span style="color:red">{start}</span>')
                response = getattr(client, "__call__")(
                    **{
                        k: v.dict() if isinstance(v, BaseModel) else v
                        for k, v in arg_values.items()
                    }
                )
            except ExecutionError as e:
                streamlit.error(e)
                return
            dialogue[-1] += response.removeprefix(start)
            streamlit.experimental_rerun()
    streamlit.button("Clear", on_click=lambda: dialogue.clear())


promt()
if len(dialogue) > MAX_CACHE:
    dialogue = dialogue[-MAX_CACHE:]

streamlit.markdown("---")
streamlit.write(
    "Built for FastAPI server at `{{server_host}}:{{server_port}}`. Docs: https://mlem.ai/doc"
)
params = urlencode({
    "text": "This app was deployed with mlem.ai!",
    "url": "https://mlem-nanogpt.fly.dev/",
    "hashtags": "mlem,mlops"
})
streamlit.components.v1.html(
    f"""<a href="https://twitter.com/share?ref_src=twsrc%5Etfw&{params}" class="twitter-share-button" data-show-count="false">
Tweet
</a>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>""")
