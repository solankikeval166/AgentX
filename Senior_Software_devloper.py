import os
import json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate, MessagesPlaceholder
import warnings


load_dotenv(dotenv_path='.env.dev')
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')

temperature = 0


def senior_software_dev(documentation):
    task_decomposer = GoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.8,
        format="json",
        api_key=GOOGLE_API_KEY
    )

    template = f"""
                You are an AI agent acting as a Senior Software Developer. Your primary responsibilities are:
                1. Assigning tasks to junior developers(in total 3 junior developers) based on a project manager's document.
                2. Determining the best technology stack for each project based on requirements.
                3. Setting deadlines for tasks to ensure timely project completion.
                
                Instructions:
                1. Read and analyze the project requirements from the project manager's document.
                2. Break down the project into smaller, manageable tasks.
                3. Assign each task to the most suitable junior developer based on their skill set.
                4. Select the most appropriate tech stack (languages, frameworks, and tools) based on project needs.
                5. Estimate realistic deadlines for task completion.
                
                Output Format:
                - Your response should be in JSON format with the following keys:
                json object containing the following keys:
                 - "developer_id": 1,
                 - "assigned_tasks": 
                 [
                 - "Implement user authentication",
                 - "Create database schema"
                ],
                 - "preferred_tech_stack": "React, Node.js, PostgreSQL",
                 
                 Project Manager's documentation:
                 {documentation}
                    """

    prompt = PromptTemplate(template=template, input_variables=["documentation"])

    requirements_chain = prompt | task_decomposer

    result = requirements_chain.invoke({"documentation": documentation})
    # return result
    
    result = result[8:-4]

    with open("sde_result.txt", "w") as f:
        f.write(result)

    return json.loads(result)


documentation = """

  "project": "Todo List Application",
  "high_level_requirements": [
    "Frontend in ReactJS",
    "Backend in NodeJS",
    "SQLite database for local storage"
  ],
  "subtasks": [
    
      "id": 1,
      "description": "Set up project structure (ReactJS & NodeJS)",
      "priority": "High",
      "instructions": "Initialize a React project using Create React App (`npx create-react-app client`). Create a NodeJS Express server directory (e.g., `server`). Install necessary packages: `express`, `sqlite3` in the server directory.  Ensure proper directory structure for separation of concerns.",
      "assignee": "Senior Software Developer"
    ,
    
      "id": 2,
      "description": "Design SQLite database schema",
      "priority": "High",
      "instructions": "Design an SQLite schema with a table for tasks including columns: `id` (INTEGER PRIMARY KEY AUTOINCREMENT), `title` (TEXT NOT NULL), `description` (TEXT), `completed` (INTEGER, default 0), `createdAt` (TIMESTAMP DEFAULT CURRENT_TIMESTAMP), `updatedAt` (TIMESTAMP DEFAULT CURRENT_TIMESTAMP).",
      "assignee": "Senior Software Developer"
    ,
    
      "id": 3,
      "description": "Implement backend API (NodeJS) - Create Task endpoint",
      "priority": "High",
      "instructions": "Create a POST endpoint (`/tasks`) to handle task creation.  Validate input data. Insert the new task into the SQLite database. Return the created task as a JSON response.",
      "assignee": "Senior Software Developer"
    ,
    
      "id": 4,
      "description": "Implement backend API (NodeJS) - Read Tasks endpoint",
      "priority": "High",
      "instructions": "Create a GET endpoint (`/tasks`) to retrieve all tasks from the database.  Allow for optional query parameters for filtering (e.g., completed status). Return tasks as a JSON array.",
      "assignee": "Senior Software Developer"
    ,
    
      "id": 5,
      "description": "Implement backend API (NodeJS) - Update Task endpoint",
      "priority": "Medium",
      "instructions": "Create a PUT endpoint (`/tasks/:id`) to update an existing task. Validate input data and ensure the task ID exists. Update the task in the database and return the updated task.",
      "assignee": "Senior Software Developer"
    ,
    
      "id": 6,
      "description": "Implement backend API (NodeJS) - Delete Task endpoint",
      "priority": "Medium",
      "instructions": "Create a DELETE endpoint (`/tasks/:id`) to delete a task. Ensure the task ID exists. Delete the task from the database and return a success response.",
      "assignee": "Senior Software Developer"
    ,
    
      "id": 7,
      "description": "Build ReactJS frontend - Task List component",
      "priority": "High",
      "instructions": "Create a component to fetch and display the list of tasks from the backend API.  Implement functionality to mark tasks as complete.  Include basic styling.",
      "assignee": "Senior Software Developer"
    ,
    
      "id": 8,
      "description": "Build ReactJS frontend - Add Task component",
      "priority": "Medium",
      "instructions": "Create a form component to allow users to add new tasks. Handle form submission and send data to the backend API.  Provide user feedback on success/failure.",
      "assignee": "Senior Software Developer"
    ,
    
      "id": 9,
      "description": "Build ReactJS frontend - Edit & Delete Task functionality",
      "priority": "Medium",
      "instructions": "Implement functionality to edit and delete existing tasks.  Handle updates and deletions via API calls.",
      "assignee": "Senior Software Developer"
    ,
    
      "id": 10,
      "description": "Integrate frontend and backend",
      "priority": "High",
      "instructions": "Connect the ReactJS frontend to the NodeJS backend API.  Ensure proper handling of API responses and error messages. Test thoroughly.",
      "assignee": "Senior Software Developer"
    ,
    
      "id": 11,
      "description": "Implement filtering and sorting",
      "priority": "Low",
      "instructions": "Add functionality to filter tasks by completion status and sort by creation date or title.",
      "assignee": "Senior Software Developer"
    ,
    
      "id": 12,
      "description": "Testing and Debugging",
      "priority": "Low",
      "instructions": "Write unit and integration tests for both frontend and backend.  Thoroughly test all API endpoints and UI interactions.",
      "assignee": "Senior Software Developer"
    
  ]

"""
print(senior_software_dev(documentation=documentation))
