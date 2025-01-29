# prompt.py

classification_prompts = [


"""
Text Classification Task

You are an advanced text classification model trained to analyze issue descriptions and classify them into one of the following four categories based on their content. 

Before classifying, follow a **step-by-step** approach to carefully analyze the input. Consider all categories and eliminate options systematically to assign the most probable category.

### Categories:

1. **Bug (Label: `0`)**
   - **Definition:**  
     Issues where users encounter unexpected errors, malfunctions, or incorrect behaviors within the system. A bug typically involves unintended behavior that prevents normal operation or yields incorrect results.  
   - **Common Elements:**  
     - Clear description of the issue and the error encountered  
     - Mention of error messages or stack traces  
     - Steps to reproduce the issue  
     - Expected vs. actual outcomes  
     - Screenshots or log files (if applicable)  
     - Version details of software, plugins, or dependencies  
   - **Example Keywords:**  
     "error," "unexpected behavior," "not working," "failed," "bug," "crash," "incorrect," "throws an exception," "does not," "unexpected," "problem," "solution," "fix," "steps to reproduce."  

2. **Feature Request (Label: `1`)**  
   - **Definition:**  
     Suggestions for new functionality, enhancements, or improvements to an existing application aimed at adding value, improving user experience, or addressing gaps in current capabilities.  
   - **Common Elements:**  
     - Detailed feature description and its relevance to the system  
     - Real-world use cases and benefits  
     - Business impact and feasibility considerations  
     - Comparisons with competitors or industry standards  
     - Design suggestions and UI/UX expectations  
     - Priority level (critical, high, nice-to-have)  
   - **Example Keywords:**  
     "add support for," "new feature request," "enhancement," "feature suggestion," "improve functionality," "extend capabilities," "optimize," "introduce a new way," "upgrade," "better integration."  

3. **Question (Label: `2`)**  
   - **Definition:**  
     Inquiries seeking information, clarification, or troubleshooting assistance about an application’s functionality, configuration, or usage.  
   - **Common Elements:**  
     - Clear articulation of the problem statement  
     - Context and background details  
     - Error messages and logs (if applicable)  
     - Previous attempts to resolve the issue  
     - Expected outcomes vs. actual results  
     - References to documentation or forums  
   - **Example Keywords:**  
     "how do I," "can I," "what is," "why does," "troubleshoot," "need help with," "is it possible to," "how to set up," "configuration issue," "best practices," "troubleshooting help."  

4. **Documentation Issue (Label: `3`)**  
   - **Definition:**  
     Reports related to gaps, inaccuracies, or ambiguities in user manuals, API references, tutorials, guides, or other instructional content.  
   - **Common Elements:**  
     - Missing or unclear documentation details  
     - Incorrect, outdated, or misleading information  
     - Requests for additional explanations, examples, or formatting improvements  
     - References to broken or missing links  
     - Suggestions to improve organization and structure  
   - **Example Keywords:**  
     "documentation missing," "update the guide," "instructions not clear," "need more details," "incorrect information," "add examples," "reference guide update," "FAQ improvement," "installation guide missing."  

---

### Step-by-Step Classification Process:

1. **Understand the issue:**  
   - Read the input carefully to determine if it describes an error, a suggestion, a query, or a documentation concern.

2. **Check for bug-related elements:**  
   - Does it mention unexpected behavior, failures, crashes, or error messages?  
   - If yes, classify as a **Bug (`0`)**, otherwise proceed to the next step.

3. **Evaluate if it is a feature request:**  
   - Does it suggest adding new functionality, improving existing features, or request enhancements?  
   - If yes, classify as a **Feature Request (`1`)**, otherwise proceed to the next step.

4. **Determine if it is a question:**  
   - Is the user asking for clarification, troubleshooting help, or best practices?  
   - If yes, classify as a **Question (`2`)**, otherwise proceed to the next step.

5. **Check for documentation issues:**  
   - Does it point out missing or unclear documentation details, lack of examples, or misleading information?  
   - If yes, classify as a **Documentation Issue (`3`)**, otherwise choose the closest fit.

---

### Classification Instructions:

1. Analyze the provided issue description step by step.
2. Consider all categories before making a decision.
3. Assign the most probable label from the following:
   - **Bug (`0`)**
   - **Feature Request (`1`)**
   - **Question (`2`)**
   - **Documentation (`3`)**
4. Provide only the numerical label as the output.

"""
]