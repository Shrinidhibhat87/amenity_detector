# Amenity Detector
This project has an end-to-end amenity detection pipeline. This was built as part of a task within a week and therefore is not the robust. Currently the pipeline is slow, susceptible to issues and needs major rework. This currently runs on streamlit with the inference being run on a local GPU.

## Instructions
1) Go over the repo, including the `Readme.md` document, the `main.py` file that is the entry point of the project.
2) Within the resource repo (`resources/excalidraw/rework_amenity_detector_draft_1.png`), there is a screenshit made using excalidraw that has some of the features that we want.
3) Dont simply agree with the decisions, rather discuss and decide on the best stack to use for the project considering costs, budget while also wanting this to be used in real life scenarios.
4) Use the `AskUserQuestionsTool` to understand requirements a bit more rather than assuming things that are not clearly written or mentioned. NOTE: The user is not an expert in all of the tools or best stack for this project, feel free to argue for a particular tool for the use case, while justifying the choice.
5) The end result is to initially come up with a SPEC.md file that we are happy about before proceeding with the actual implementation. Also suggest the implementation in phases

## Requirements in terms of code/standards to follow
1) Currently we use pip install mechanism to recreate the project in other machines. We want to transition to uv packages. Please make that happen
2) Any code that would be later written, needs to have proper comments, return types, datatypes etc as the user needs to be able to understand the code considering the user is NOT having a huge amount of experience
3) The user wants to have the option to play around with three VLMS. The Candidate VLMs are:
	1) Qwen2.5-VL-7B Instruct
	2) Gemma 3 (Gemini 2.0 Flash is free upto 1500 requests a day)
	3) LLaMA 3.2 Vision

    Please create a way to make this happen where even local deployment should work
4) The goal of this project is to make it have the end-to-end feeling of how a real product is, therefore the stack needs to have all the components (MLOps: Docker compose, manifests, etc, model deployment etc). Along with that goal, the other goal is for the user to also use this as a learning curve to understand various aspects of AI Engineering.

DO NOT assume things. Ask clarity from the users, while also suggesting the best practises and best tools to use considering the budget.