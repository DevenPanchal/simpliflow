#!/usr/bin/env python
# coding: utf-8

import requests
import json
import importlib.util
import sys
import func
import os


from openai import OpenAI
import anthropic
import cohere

class Agent:
    def __init__(self, name):
        self.messages = []
        self.name = name

    # Function that takes a user question as input and returns the chatbot's response
    def get_response(
        self, question, vendor, modelname, creativity, diversity, maxtokens
    ):

      
        # If the values are None, then use the default values
        if vendor is None:
            vendor = "OpenAI"

        if modelname is None:
            modelname = "gpt-4o"

        if creativity is None:
            creativity = 0.7

        if diversity is None:
            diversity = 0.95

        if maxtokens is None:
            maxtokens = 2000

        if vendor == "OpenAI":

            

            client = OpenAI()

            completion = client.chat.completions.create(
                model=modelname,
                messages=self.messages,
                temperature=creativity,
                top_p=diversity,
                max_tokens=maxtokens,  # max_tokens will is deprecated by OpenAI. It will be removed/replaced in future versions.
            )

            output = completion.choices[0].message.content

            return output

        if vendor == "Anthropic":

            

            client = anthropic.Anthropic()

            # extract system message from the first message

            message = client.messages.create(
                model=modelname,
                max_tokens=maxtokens,
                temperature=creativity,
                system=self.messages[0][
                    "content"
                ],  # first message of our list is system message
                messages=self.messages[1:],  # rest of the messages are user messages,
                top_p=diversity,
            )

            output = message.content[0].text
            return output
        
        if vendor == "Cohere":
            
            

            # Import the API key from the environment variables
            api_key = os.getenv('COHERE_API_KEY')

            co = cohere.ClientV2(api_key)

    
            output = co.chat(
                model=modelname,
                messages=self.messages,
                max_tokens=maxtokens,
                temperature=creativity,
                p = diversity

            )
           
            return output.dict()['message']['content'][0]['text']
        
        
        if vendor == "DeepSeek":

            
            client = OpenAI(api_key=os.getenv('DEEPSEEK_API_KEY'), base_url="https://api.deepseek.com")

            response = client.chat.completions.create(
                model=modelname,
                messages=self.messages,
                temperature=creativity,
                top_p=diversity,
                max_tokens=maxtokens,  # max_tokens will is deprecated by OpenAI. It will be removed/replaced in future versions.
        
            )

            output = response.choices[0].message.content
            return output
        
        if vendor == "Google":
      
            client = OpenAI(api_key=os.getenv('GEMINI_API_KEY'), base_url="https://generativelanguage.googleapis.com/v1beta/openai/")

            response = client.chat.completions.create(
            model=modelname,
            messages=self.messages,
            temperature=creativity,
            top_p=diversity,
            max_tokens=maxtokens,  # max_tokens will is deprecated by OpenAI. It will be removed/replaced in future versions.
        
        )   
            output = response.choices[0].message.content
            return output



def call_agents(
    agentsfile, dynamic_input, vendor, modelname, creativity, diversity, maxtokens
):

    resultofpreviousagent = None
    interactions = []  # Holds all Interactions in the chain.

    with open(agentsfile) as f:
        data = json.load(f)
        # continue extracting agents from the JSON file until next_agent is not found
        for agent in data["agents"]:

            # head and next is also not used

            # Instantiate the Agent class with the above parameters
            # name not used currently, but just passed to the instance
            agent_instance = Agent(agent["name_of_agent"])

            # if the agent is the first agent in the chain, the question is the input  else the question is the result of the previous agent
            if agent["head"] == "True":
                question = agent["what_should_agent_do"]

                # if dynamic_input is not None, append it to the question, so it is passed to the agent
                if dynamic_input is not None:
                    question = (
                        agent["what_should_agent_do"]
                        + "- The input is : "
                        + dynamic_input
                    )
                # question = 'Do the task given to you.'
                # question = data['input']
            else:
                question = (
                    agent["what_should_agent_do"]
                    + "---"
                    + resultofpreviousagent
                    + "---"
                )

                # question = resultofpreviousagent

            # log startup messages for the agent to agent_instance.messages,  interactions
            agent_instance.messages = [
                {"role": "system", "content": "You are a " + agent["role_of_agent"]}
            ]
            agent_instance.messages.append({"role": "user", "content": question})

            # make the call
            resultofagent = agent_instance.get_response(
                question, vendor, modelname, creativity, diversity, maxtokens
            )

            # log responses, results to agent_instance.messages, interactions and resultofpreviousagent
            resultofpreviousagent = resultofagent
            agent_instance.messages.append({"role": "system", "content": resultofagent})

            interactions.append(
                {
                    "entity": "agent",
                    "name": agent["name_of_agent"],
                    "input": question,
                    "output": resultofagent,
                }
            )

            # print messages
            print(
                "################################# Agent Name:",
                agent["name_of_agent"],
                "#################################",
            )
            # logging.debug('################################# Agent Name:', agent['name_of_agent'], '#################################')
            print(resultofagent, end="\n\n")
            # logging.debug(resultofagent)

            # if postprocessor_function is not None, call the function with the result of the agent
            if agent["postprocessor_function"] != "None":

                postprocessor_function = agent["postprocessor_function"]
                """
                print (globals())
            
                
                fullnamepostprocessor_function = 'func.'+ postprocessor_function

                # call the function dynamically using the function name
                postprocessor_function_obj = globals()[fullnamepostprocessor_function]
                resultoffunction = postprocessor_function_obj(resultofpreviousagent)
                """

                r = getattr(func, postprocessor_function)
                resultoffunction = r(resultofpreviousagent)

                # store the modified result in resultofpreviousagent
                resultofpreviousagent = resultoffunction
                print(
                    "################################# Postprocessor Function:",
                    postprocessor_function,
                    "#################################",
                )
                # logging.debug('################################# Postprocessor Function:', postprocessor_function, '#################################')
                print(resultofpreviousagent, end="\n\n")
                # logging.debug(resultofpreviousagent)

                # also store the result of function in interactions
                interactions.append(
                    {
                        "entity": "function",
                        "name": postprocessor_function,
                        "input": resultofagent,
                        "output": resultoffunction,
                    }
                )

    # Write the interactions to a JSON file. The location of the Interactions file will be inside an Interactions folder. The name of the interactions file will be borrowed from the agentsfile identifier.
    filename = os.path.splitext(os.path.basename(agentsfile))[0] + "_interactions.json"
    filepath = os.path.join("Interactions", filename)

    # If the Interactions folder does not exist, create it
    if not os.path.exists("Interactions"):
        os.makedirs("Interactions")

    with open(filepath, "w") as f:
        json.dump(interactions, f, indent=4)

    return resultofpreviousagent, interactions
