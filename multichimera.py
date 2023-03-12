import copy
import math
import os
import random
import sys
import traceback
import shlex

import modules.scripts as scripts
import gradio as gr

from modules import sd_samplers
from modules.processing import Processed, process_images
from PIL import Image
from modules.shared import opts, cmd_opts, state


def process_string_tag(tag):
    return tag


def process_int_tag(tag):
    return int(tag)


def process_float_tag(tag):
    return float(tag)


def process_boolean_tag(tag):
    return True if (tag == "true") else False

def cmdargs(line):
    args = shlex.split(line)
    pos = 0
    res = {}

    while pos < len(args):
        arg = args[pos]

        assert arg.startswith("--"), f'must start with "--": {arg}'
        assert pos+1 < len(args), f'missing argument for command line option {arg}'

        tag = arg[2:]

        if tag == "prompt" or tag == "negative_prompt":
            pos += 1
            prompt = args[pos]
            pos += 1
            while pos < len(args) and not args[pos].startswith("--"):
                prompt += " "
                prompt += args[pos]
                pos += 1
            res[tag] = prompt
            continue


        func = prompt_tags.get(tag, None)
        assert func, f'unknown commandline option: {arg}'

        val = args[pos+1]
        if tag == "sampler_name":
            val = sd_samplers.samplers_map.get(val.lower(), None)

        res[tag] = func(val)

        pos += 2

    return res


prompt_tags = {
    "sd_model": None,
    "outpath_samples": process_string_tag,
    "outpath_grids": process_string_tag,
    "prompt_for_display": process_string_tag,
    "prompt": process_string_tag,
    "negative_prompt": process_string_tag,
    "styles": process_string_tag,
    "seed": process_int_tag,
    "subseed_strength": process_float_tag,
    "subseed": process_int_tag,
    "seed_resize_from_h": process_int_tag,
    "seed_resize_from_w": process_int_tag,
    "sampler_index": process_int_tag,
    "sampler_name": process_string_tag,
    "batch_size": process_int_tag,
    "n_iter": process_int_tag,
    "steps": process_int_tag,
    "cfg_scale": process_float_tag,
    "width": process_int_tag,
    "height": process_int_tag,
    "restore_faces": process_boolean_tag,
    "tiling": process_boolean_tag,
    "do_not_save_samples": process_boolean_tag,
    "do_not_save_grid": process_boolean_tag
}

class Script(scripts.Script):
    def title(self):
        return "MultiChimera Beta"

    def ui(self, is_img2img):    
        gr.HTML('<br />')
        steps_override = gr.Slider(label="Total Steps", min=20, max=150, value=80, step=1, interactive=True, elem_id=self.elem_id("steps_override"))
        #This steps slider will override whatever steps were entered in the original UI
        universal_prompt = gr.Textbox(label="Universal Prompt (use * character where the prompt edits should be placed)", lines=2, elem_id=self.elem_id("universal_prompt"))
        #Universal Prompt is above - Do not mess wif dat lol.
        with gr.Row():
            with gr.Column():
            #Here are all textboxes for up to 10x Chimeras. Yes, a dropdown would be nice, but this shit is hard enough as it is, OK?
            #We are bruteforcing the prompt to default any empty box to "999". That essentially nullifies them from being taken into account in the final prompt.
                p1 = gr.Textbox(label='Subject #1', lines=1, elem_id=self.elem_id("p1"))
                p2 = gr.Textbox(label='Subject #2', lines=1, elem_id=self.elem_id("p2"))
                p3 = gr.Textbox(label='Subject #3', lines=1, elem_id=self.elem_id("p3"))
                p4 = gr.Textbox(label='Subject #4', lines=1, elem_id=self.elem_id("p4"))
                p5 = gr.Textbox(label='Subject #5', lines=1, elem_id=self.elem_id("p5"))
                p6 = gr.Textbox(label='Subject #6', lines=1, elem_id=self.elem_id("p6"))
                p7 = gr.Textbox(label='Subject #7', lines=1, elem_id=self.elem_id("p7"))
                p8 = gr.Textbox(label='Subject #8', lines=1, elem_id=self.elem_id("p8"))
                p9 = gr.Textbox(label='Subject #9', lines=1, elem_id=self.elem_id("p9"))
                p10 = gr.Textbox(label='Subject #10', lines=1, elem_id=self.elem_id("p10"))
            with gr.Column():
                p1f = gr.Number(label="From Step:", elem_id=self.elem_id("p1f")) #This one will always be 0, so having it as an input is irrelevant.
                p2f = gr.Number(label="From Step:", elem_id=self.elem_id("p2f"))
                p3f = gr.Number(label="From Step:", elem_id=self.elem_id("p3f"))
                p4f = gr.Number(label="From Step:", elem_id=self.elem_id("p4f"))
                p5f = gr.Number(label="From Step:", elem_id=self.elem_id("p5f"))
                p6f = gr.Number(label="From Step:", elem_id=self.elem_id("p6f"))
                p7f = gr.Number(label="From Step:", elem_id=self.elem_id("p7f"))
                p8f = gr.Number(label="From Step:", elem_id=self.elem_id("p8f"))
                p9f = gr.Number(label="From Step:", elem_id=self.elem_id("p9f"))
                p10f = gr.Number(label="From Step:", elem_id=self.elem_id("p10f")) #This one does not need to be stated, since prompt editing does not require one to tell it to generate until the last step. It will do so by itself.
            with gr.Row():
                with gr.Column():
                    p1t = gr.Number(label="To Step:", min_width=40, elem_id=self.elem_id("p1t"))
                    p2t = gr.Number(label="To Step:", min_width=80, elem_id=self.elem_id("p2t"))
                    p3t = gr.Number(label="To Step:", elem_id=self.elem_id("p3t"))
                    p4t = gr.Number(label="To Step:", elem_id=self.elem_id("p4t"))
                    p5t = gr.Number(label="To Step:", min_width=40, elem_id=self.elem_id("p5t"))
                    p6t = gr.Number(label="To Step:", min_width=80, elem_id=self.elem_id("p6t"))
                    p7t = gr.Number(label="To Step:", elem_id=self.elem_id("p7t"))
                    p8t = gr.Number(label="To Step:", elem_id=self.elem_id("p8t"))
                    p9t = gr.Number(label="To Step:", elem_id=self.elem_id("p9t"))
                    p10t = gr.Number(label="To Step:", elem_id=self.elem_id("p10t"))


#This is essentially the final chimera prompt up to the maximum value of 10 chimeras

        return [universal_prompt, steps_override, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p1t, p2t, p3t, p4t, p5t, p6t, p7t, p8t, p9t]

    def run(self, p, universal_prompt, steps_override, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p1t, p2t, p3t, p4t, p5t, p6t, p7t, p8t, p9t):

        #Replace unused boxes and zeroes for 999, hey, it's not fancy but it works.
        if p1 == "" or p1 == "0":
            p1 = "nothing"
        if p2 == "" or p2 == "0":
            p2 = "nothing"
        if p3 == "" or p3 == "0":
            p3 = "nothing"
        if p4 == "" or p4 == "0":
            p4 = "nothing"
        if p5 == "" or p5 == "0":
            p5 = "nothing"
        if p6 == "" or p6 == "0":
            p6 = "nothing"
        if p7 == "" or p7 == "0":
            p7 = "nothing"
        if p8 == "" or p8 == "0":
            p8 = "nothing"
        if p9 == "" or p9 == "0":
            p9 = "nothing"
        if p10 == "" or p10 == "0":
            p10 = "nothing"

        if p1t == "" or p1t == 0:
            p1t = 999
        if p2t == "" or p2t == 0:
            p2t = 999
        if p3t == "" or p3t == 0:
            p3t = 999
        if p4t == "" or p4t == 0:
            p4t = 999
        if p5t == "" or p5t == 0:
            p5t = 999
        if p6t == "" or p6t == 0:
            p6t = 999
        if p7t == "" or p7t == 0:
            p7t = 999
        if p8t == "" or p8t == 0:
            p8t = 999
        if p9t == "" or p9t == 0:
            p9t = 999

        #This is the 10x chimera master prompt that works with anything from 2 to 10 chimeras.
        final_prompt = "[[[[[[[[[" + str(p1) + ":" + str(p2) + ":" + str(int(p1t)) + "]" + ":" + str(p3) + ":" + str(int(p2t)) + "]:" + str(p4) + ":" + str(int(p3t)) + "]:" + str(p5) + ":" + str(int(p4t)) + "]:" + str(p6) + ":" + str(int(p5t)) + "]:" + str(p7) + ":" + str(int(p6t)) + "]:" + str(p8) + ":" + str(int(p7t)) + "]:" + str(p9) + ":" + str(int(p8t)) + "]:" + str(p10) + ":" + str(int(p9t)) + "]"

        #bad naming for sure but all we doing here is replacing the star in the universal prompt with our final chimera prompt and voila! Bon appettit. 
        grandmaster_prompt = universal_prompt.replace("*", final_prompt)

        #idk what this bullshit is l0l I just stole it from a working script but hey it werks!! sort of. It doesn't output the final prompt as it usually does for regular txt2img. Plz halp?
        p.do_not_save_grid = True

        job_count = 0
        jobs = []

        args = {"prompt": grandmaster_prompt} #memo to myself... do the dumb things I gotta do... check to see if negative prompts work fine without me having to do anything 'round here.

        job_count += args.get("n_iter", p.n_iter)

        jobs.append(args)

        p.seed = int(random.randrange(4294967294)) #if we can find a way to save the seed so we can reuse the same seed, idk, I'll try removing this shit and see if the code works fine without it.

        p.steps = int(steps_override)

        state.job_count = job_count

        images = []
        all_prompts = []
        infotexts = []
        for n, args in enumerate(jobs):
            state.job = f"{state.job_no + 1} out of {state.job_count}"

            copy_p = copy.copy(p)
            for k, v in args.items():
                setattr(copy_p, k, v)

            proc = process_images(copy_p)
            images += proc.images

        return Processed(p, images, p.seed, "", all_prompts=all_prompts, infotexts=infotexts)

