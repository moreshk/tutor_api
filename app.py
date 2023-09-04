from flask import Flask, render_template, request, jsonify
import json
import os
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

# Fetch the OPENAI_API_KEY from environment variables
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY in the .env file.")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI


# This is an LLMChain to create a task for a particular exam

llm = ChatOpenAI(temperature=1, model_name="gpt-3.5-turbo")
template = """You are a test task creator for {test_type}. I will provide you the test type and your job is to create a task for a test taker.

You will provide a topic and ask the user if they agree or disagree, provide pros and cons etc and provide their views on the topic.

Sample Tasks for {test_type}
Sample 1: Technology
Task: "Some people believe that technological advancements lead to the loss of traditional cultures. To what extent do you agree or disagree?"

Sample 2: Education
Task: "Some educators argue that every child should be taught how to play a musical instrument. Discuss the advantages and disadvantages of this. Give your own opinion."

Sample 3: Environment
Task: "Climate change is now an accepted threat to our planet, but there is not enough political action to control excessive consumerism and pollution. Discuss both views and give your own opinion."

Sample 4: Health
Task: "Some people think that governments should focus on reducing healthcare costs, rather than funding arts and sports. Do you agree or disagree?"

Sample 5: Society
Task: "Some people think that the best way to reduce crime is to give longer prison sentences. Others, however, believe there are better alternative ways of reducing crime. Discuss both views and give your opinion."

Sample 6: Work
Task: "Remote work is becoming increasingly popular. Discuss the advantages and disadvantages of working from home."

Sample 7: Global Issues
Task: "Some people argue that developed countries have a higher obligation to combat climate change than developing countries. Discuss both sides and give your own opinion."

Sample 8: Science
Task: "Genetic engineering is an important issue in modern society. Some people think that it will improve people’s lives in many ways. Others feel that it may be a threat to life on earth. Discuss both these views and give your own opinion."

You will use the above examples only as a guideline for framing the task and create a new task and description randomly on a different topic. No need to use the word Sample in the task description. 

Create 5 such tasks and descriptions based on the above guidelines. In your output mention these ten tasks and format the output as Title:  and Description: .
"""
prompt_template = PromptTemplate(input_variables=["test_type"], template=template)
task_creator_chain = LLMChain(llm=llm, prompt=prompt_template)

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

template = """You are a test task selector. I will provide you a list of five tasks and their respective descriptions below. 

{five_tasks}

You will select one of the tasks randomly and output it along with its description in a json format that has the following fields:

"title": "picked randomly from the ten tasks",
"description": "description for the randomly picked title"
"""

prompt_template = PromptTemplate(input_variables=["five_tasks"], template=template)
task_selector_chain = LLMChain(llm=llm, prompt=prompt_template)

# print(task_selector_chain.run(writing_task))

# This is the overall chain where we run these two chains in sequence.
from langchain.chains import SimpleSequentialChain
overall_chain = SimpleSequentialChain(chains=[task_creator_chain, task_selector_chain], verbose=True)

# print(overall_chain.run("IELTS Academic Writing Task 2"))


app = Flask(__name__)


# API endpoint for generating a test
@app.route('/api/generate_test', methods=['POST'])
def api_generate_test():
    generated_data = overall_chain.run("IELTS Academic Writing Task 2")
    try:
        generated_data = json.loads(generated_data)
    except (json.JSONDecodeError, TypeError):
        pass

    if isinstance(generated_data, dict):
        title = generated_data.get('title', 'Default Title')
        description = generated_data.get('description', 'Default Description')
        test_data = {
            'title': title,
            'description': description
        }
        return jsonify(test_data)
    else:
        return jsonify({'error': 'Unexpected data format'})

# API endpoint for grading a response
@app.route('/api/grade_response', methods=['POST'])
def api_grade_response():
    data = request.json
    user_response = data.get('userResponse')
    title = data.get('title')
    description = data.get('description')

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    prompt = PromptTemplate(
        input_variables=["essay", "task_title", "task_desc"],
        template="""You are an essay grader for IELTS Writing Task 2. Your criteria for grading is how well the provided essay fulfils the requirements of the task, 
        including whether it has addressed all parts of the task description and provided a clear position. 
        The key points for your grading are:
        1. Does the essay address all parts of the prompt (task and task description).
        2. Provides a clear thesis statement that outlines the writers position.
        3. Support the writers arguments with relevant examples and evidence.
        
        Task Title: {task_title}

        Task Description: {task_desc}

        Essay: {essay}

        Grade the essay out of 10 on each of the 3 points. Provide detailed description about how much you graded the essay on each of the points and provide feedback on how it could improve. Finally provide the average grade based on the 3 grades.
        """,
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    inputs = {
        "essay": user_response,
        "task_title": title,
        "task_desc": description
    }

    # print(chain.run(inputs))

    # Your existing logic for grading
    feedback_from_api = chain.run(inputs)  # Replace with actual feedback

    return jsonify({'feedback': feedback_from_api})

# API endpoint for grading a response for coherence
@app.route('/api/grade_response_coherence', methods=['POST'])
def api_grade_response_coherence():
    data = request.json
    user_response = data.get('userResponse')
    title = data.get('title')
    description = data.get('description')

    # Your existing logic for grading
    # Your code to generate the grade and feedback using OpenAI API
    # For example:
    # grade, feedback = your_function_to_generate_grade(user_response, title, description)

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    prompt = PromptTemplate(
        input_variables=["essay", "task_title", "task_desc"],
        template="""You are an essay grader for IELTS Writing Task 2. Your criteria for grading is how well the provided essay is in terms of its coherence and cohesion in context of the title and description of the task.
        You will evaluate the organization and flow of the essay. Look at how well the ideas are sequenced and how well the paragraphs are linked.
        The key points for your grading are:
        1. Use clear paragraphing with topic sentences.
        2. Use cohesive devices (e.g., furthermore, however, in addition) effectively.
        3. Make sure your ideas are logically organized and easy to follow.
        
        Task Title: {task_title}

        Task Description: {task_desc}

        Essay: {essay}

        Grade the essay out of 10 on each of the 3 points. Provide detailed description about how much you graded the essay on each of the points and provide feedback on how it could improve. Finally provide the average grade based on the 3 grades.
        """,
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    inputs = {
        "essay": user_response,
        "task_title": title,
        "task_desc": description
    }

    feedback_from_api = chain.run(inputs)  # Replace with actual feedback

    return jsonify({'feedback': feedback_from_api})

# @app.route('/grade_response_lexical', methods=['POST'])
# def grade_response_lexical():
#     # Your code here, similar to grade_response but with different grading logic
#     return jsonify({'feedback': feedback_from_api})

# @app.route('/grade_response_grammatical', methods=['POST'])
# def grade_response_grammatical():
#     # Your code here, similar to grade_response but with different grading logic
#     return jsonify({'feedback': feedback_from_api})


if __name__ == '__main__':
    app.run(debug=True)

