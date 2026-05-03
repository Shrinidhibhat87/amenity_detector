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

## Updated Work needed

## Redesigning and making things work for the Amenity Detector

### Premise
You already know the history of the project. Currently when trying to run this application, the images are being uploaded, but the description is not being generated.
We can not even see what the error is, because the print that we see is of an exception when trying to execute the generate command in all 3 options.
We cannot see what the error actually is. Screenshot: (`resources/screenshot_failure.png`)

What are we doing when deploying this?
On WSL, we have one terminal that has "ollama serve". In another we have a "docker compose up --build" command executed.
We see that the images are being saved in the second screenshot (`resources/browse_properties.png`).
Therefore we believe this could be an issue with the deployment or something in our model generation code.
The path of remote usage of Gemini as well did not work as you can see.

### What do we need from you
Q1) Please help us understand why do we need to use Prometheus and Grafana? What value do they give and how is this is used in real life industry?

T1) We need to fix the above generation part. Please identify the issue and help fix this.

T2) The current design of the website is very somber and not appealing. We need to change this. We need to have a homepage similar to `resources/website_page_1.png`.
This is a screenshot, but the general idea (without the building image behind), we want something with light colors and appealing to the audience.
The goal/idea here is to allow users to get an amenity detector and descriptor readily available. 
1) The upload and generate button should take us the page similar to what you have made. NOTE: Please improve the coloring and tone. Here they should be able to upload images,
	get the list of amenities for each room. The user should be allowed to change this as well with an option somewhere with an edit button.
	The descriptor should describe the property and highlight the positives of the place.
	It should also clearly mention the various rooms available. It should be able to classify kitchen, living room, bedroom etc.
	NOTE: All of this needs to be editable by the user.
	a) Once the user uploads, we detect the various rooms. Then we detect the amenities in each room. After this we list them down to the user.
	b) The user can then edit the table if they want, when confirming or rejecting the presence of that.
	c) After edits and changes, the user can hit confirm, which will then go and generate a description of the property based on the revised list of rooms and amenities.
	
2) The Browse property feature should be implemented in the next phase. This is where we want to integrate a voice assistant that will take in the users choices/requests and then recommend them the right place present in the DB


T3) Once you have made the above changes, list down the steps that I need to do to test this new feature. If the test passes, we move on to the git related concepts.


### Important points
1) Use the /AskUserQuestionTool and your superpowers to get the best possible design and implementation.
2) Do not assume stuff, instead ask questions of clarification.
3) Ensure the code that you generate is well tested
4) Continue to update the readme.md file