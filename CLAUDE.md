# Amenity Detector
This project has an end-to-end amenity detection pipeline. This was built as part of a task within a week and therefore is not the robust. Currently the pipeline is slow, susceptible to issues and needs major rework. This currently runs on streamlit with the inference being run on a local GPU.

## Instructions
1) Go over the repo, including the `Readme.md` document, the `main.py` file that is the entry point of the project.
2) Within the resource repo (`resources/excalidraw/rework_amenity_detector_draft_1.png`), there is a screenshit made using excalidraw that has some of the features that we want.
3) Dont simply agree with the decisions, rather discuss and decide on the best stack to use for the project considering costs, budget while also wanting this to be used in real life scenarios.
4) Use the `AskUserQuestionsTool` to understand requirements a bit more rather than assuming things that are not clearly written or mentioned. NOTE: The user is not an expert in all of the tools or best stack for this project, feel free to argue for a particular tool for the use case, while justifying the choice.
5) The end result is to initially come up with a SPEC.md file that we are happy about before proceeding with the actual implementation. Also suggest the implementation in phases

## Requirements in terms of code/standards to follow
1) The goal of this project is to make it have the end-to-end feeling of how a real product is, therefore the stack needs to have all the components (MLOps: Docker compose, manifests, etc, model deployment etc). Along with that goal, the other goal is for the user to also use this as a learning curve to understand various aspects of AI Engineering.

DO NOT assume things. Ask clarity from the users, while also suggesting the best practises and best tools to use considering the budget.

### Important points
1) Use the /AskUserQuestionTool and your superpowers to get the best possible design and implementation.
2) Do not assume stuff, instead ask questions of clarification.
3) Ensure the code that you generate is well tested. Test driven development
4) Continue to update the readme.md file