import os
import json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate, MessagesPlaceholder
import warnings


load_dotenv(dotenv_path='.env.dev')
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')

temperature = 0


def po_instructions(defination, requirements):
    requirements_finder = GoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.8,
        format="json",
        api_key=GOOGLE_API_KEY
    )

    template = f"""
                You are a Product Owner AI agent with expertise in Agile project management and backlog prioritization. 
                
                Your responsibilities include:
                    1. Goal Analysis and Prioritization: 
                     - Review input goals provided by the business analyst. Assess their relevance, urgency, and alignment with the product vision. 
                     - Assign a priority level (e.g., High, Medium, Low) to each goal based on business impact and feasibility.

                    2. Backlog Management: Maintain a well-organized product backlog by categorizing tasks, features, and improvements. 
                     - Ensure all items are updated with clear descriptions, priorities, and status (e.g., To Do, In Progress, Completed).
                     - Instruction and Output: Summarize the top-priority goals into a JSON object formatted as follows:
                        "priority_goals": [
                            "id": "unique_goal_id",
                            "title": "Goal Title",
                            "description": "Detailed description of the goal",
                            "priority": "High/Medium/Low",
                            "status": "To Do/In Progress/Completed"
                        ]
                
                    3. Project Manager Communication: Provide clear instructions to the project manager by listing the main priority goals along with recommended next steps to address them.
                     - Input Example:
                        - Goals from business analysts (e.g., "Improve website UX", "Launch feature X by Q2").
                        - Current product backlog items.
                        
                     - Output Example:
                       - JSON structure of priority goals as described above.
                       - A summary message to the project manager:
                         - "The top three goals for this sprint are defined in the JSON output. 
                            Please prioritize resources to address these items by [specific deadline]. Begin by [specific recommendation]."
                
                Example output:
                "project_name":"To-Do List Application"
                "priority_goals": [
                            "id": "1",
                            "title": "Build frontend",
                            "description": "Please use ReactJS to build frontend application and ensure that it is responsive to all devices.",
                            "priority": "High",
                            "status": "To Do"
                        ] 
                "priority_goals": [
                            "id": "2",
                            "title": "Build Backend",
                            "description": "Please use NodeJS to build backend server and ensure that it is scalable enough for multiple users to all devices.",
                            "priority": "High",
                            "status": "To Do"
                        ] 
                
                Constraints:
                Always align with Agile principles.
                Outputs must be actionable, concise, and prioritized effectively.

                Generate an actionable response based on the following input data:
                Here is the project defination to be implemented.
                {defination}
                
                and analyzed project goals and neccessary things by buisness analyst:
                {requirements}
                Provide sole output as json no preemble explaination required.
                """
                
    prompt = PromptTemplate(template=template, input_variables=["defination","requirements"])

    requirements_chain = prompt | requirements_finder 

    result = requirements_chain.invoke({"defination": defination, "requirements": requirements})
    
    start_idx = result.find("{")
    end_idx = result.rfind("}") + 1
    result = result[start_idx:end_idx]
    with open("pm_result.txt", "w") as f:
        f.write(result)

    return json.loads(result)


requirements = """

  "project_name": "Todo List Application",
  "goals": [
    "Build a todo list application using ReactJS and NodeJS",
    "Utilize a local SQLite database for data persistence"
  ],
  "functional_requirements": [
    "Add new tasks with descriptions and due dates",
    "Mark tasks as complete",
    "Edit existing tasks",
    "Delete tasks",
    "Filter tasks by status (complete/incomplete)",
    "Sort tasks by due date or creation date",
    "View a list of all tasks",
    "Persist data locally using SQLite"
  ],
  "non_functional_requirements": [
    "User-friendly interface",
    "Responsive design for different screen sizes",
    "Fast loading times",
    "Data persistence should be reliable and consistent",
    "Application should be stable and error-free"
  ],
  "technical_requirements": [
    "Frontend development using ReactJS",
    "Backend development using NodeJS",
    "Local SQLite database integration",
    "API for communication between frontend and backend",
    "Appropriate testing framework for both frontend and backend"
  ],
  "stakeholder_requirements": 
    "Developer": [
      "Use of established best practices for ReactJS and NodeJS development",
      "Well-documented codebase",
      "Easy to maintain and extend application"
    ]
  

"""
print(po_instructions("I want to build todo list application using ReactJS and NodeJS. My application will have SQLite database stored locally.",requirements=requirements))