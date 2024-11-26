# LiteLLM API engine, designed to largely mimic the OpenAIChatCompletions Format

import copy
import os
from collections import defaultdict
from importlib.util import find_spec
from typing import List, Literal, Optional, Tuple
import time

from tqdm import tqdm
from IPython import embed

import lm_eval.models.utils
from lm_eval import utils
from lm_eval.api.model import LM, TemplateLM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import retry_on_specific_exceptions
from lm_eval.utils import eval_logger

import litellm
litellm.vertex_project = "temp-project" # TODO: replace with your own project name
litellm.vertex_location = "us-central1"  # proj location

@register_model("litellm")
class LiteLLM(LM):
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        truncate: bool = False,
        **kwargs,
    ) -> None:
        """

        :param model: str
            Implements an OpenAI-style chat completion API for
            models via LiteLLM
            HuggingFace Tokenizer
            OpenAI API model (e.g. gpt-3.5-turbo)
            using the **gen_kwargs passed on init
        :param truncate: bool
            Truncate input if too long (if False and input is too long, throw error)
        """
        super().__init__()
        self.model = model
        self.truncate = truncate

    @property
    def max_length(self) -> int:
        # Note: the OpenAI API supports up to 2049 tokens, with the first token being the first input token
        return 4096

    @property
    def max_gen_toks(self) -> int:
        return 4096

    @property
    def batch_size(self):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    @property
    def device(self):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    def generate_until(self, requests) -> List[str]:
        res = defaultdict(list)
        re_ords = {}

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        grouper = lm_eval.models.utils.Grouper(requests, lambda x: str(x.args[1]))
        for key, reqs in grouper.get_grouped().items():
            # within each set of reqs for given kwargs, we reorder by token length, descending.
            re_ords[key] = utils.Reorderer(
                [req.args for req in reqs], lambda x: (-len(x[0]), x[0])
            )

        pbar = tqdm(total=len(requests), disable=(self.rank != 0))
        for key, re_ord in re_ords.items():
            # n needs to be 1 because messages in
            # chat completion are not batch but
            # is regarded as a single conversation.
            chunks = lm_eval.models.utils.chunks(re_ord.get_reordered(), n=1)
            for chunk in chunks:
                contexts, all_gen_kwargs = zip(*chunk)
                inps = [{"role": "user", "content": context} for context in contexts]

                gen_kwargs = all_gen_kwargs[0]
                until = None
                if isinstance(kwargs := copy.deepcopy(gen_kwargs), dict):
                    if "do_sample" in kwargs.keys():
                        kwargs.pop("do_sample")
                    if "until" in kwargs.keys():
                        until = kwargs.pop("until")
                        if isinstance(until, str):
                            until = [kwargs]
                        elif not isinstance(until, list):
                            raise ValueError(
                                f"Expected repr(kwargs['until']) to be of type Union[str, list] but got {until}"
                            )
                        kwargs["stop"] = until
                    kwargs["max_tokens"] = kwargs.pop("max_gen_toks", self.max_gen_toks)
                else:
                    raise ValueError(
                        f"Expected repr(kwargs) to be of type repr(dict) but got {kwargs}"
                    )
                
                if "mistral" in self.model or "gemini" in self.model:
                    # Mistral doesn't support the stop parameter, so remove
                    kwargs.pop("stop", None)
                
                if "gemini" in self.model:
                    # Sleep between requests for rate limit
                    time.sleep(3)
                
                try:
                    response = litellm.completion(num_retries=10, 
                        messages=inps,
                        model=self.model,
                        **kwargs,
                    )

                    for resp, (context, args_) in zip(response.choices, chunk):
                        s = resp.message.content

                        if until is not None:
                            for term in until:
                                if len(term) > 0:
                                    s = s.split(term)[0]

                        res[key].append(s)

                        self.cache_hook.add_partial(
                            "generate_until", (context, {"until": until}), s
                        )
                        pbar.update(1)
            
                except Exception as e:
                    print("Error in generate_until: {}".format(str(e)))
                    res[key].append("0")

            # reorder this group of results back to original unsorted form
            res[key] = re_ord.get_original(res[key])

        pbar.close()

        return grouper.get_original(res)

    def loglikelihood(self, requests):
        raise NotImplementedError("No support for logits.")

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError("No support for logits.")