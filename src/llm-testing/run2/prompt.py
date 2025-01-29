# prompt.py

classification_prompts = [


"""

Text Classification Task

You are an advanced text classification model trained to analyze issue descriptions and classify them into one of the following four categories based on their content:

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
   - **Examples:**  
     1. TypeError Example LOGO. I’m using moviepy 1.0.3, it’s amazing. I tried creating a logo following the official example, but encountered multiple issues such as "TypeError: FUNCTION got an unexpected keyword argument 'mask'."  
        Output: 0  

     2. Application crashes when opening file. After the latest update, the application crashes when trying to open a .csv file. No error message is shown, just a sudden exit.  
        Output: 0  

     3. UI layout broken on mobile view. The navigation bar overlaps with the content in the mobile version, making it impossible to click buttons. Expected a responsive design.  
        Output: 0  

     4. Memory leak in background process. Observed increased memory usage over time when running the background sync feature. After a few hours, the process consumes over 90% of RAM.  
        Output: 0  

     5. Authentication not working with special characters. When logging in with special characters in the password, authentication fails with a "400 Bad Request" error.  
        Output: 0  

2. **Feature Request (Label: `1`)**  
   - **Definition:**  
     Suggestions for new functionality, enhancements, or improvements to an existing application aimed at adding value, improving user experience, or addressing gaps in current capabilities.  
   - **Examples:**  
     1. Add dark mode feature. A dark mode would improve the user experience for those working in low-light environments.  
        Output: 1  

     2. Export reports to PDF. It would be helpful to have an option to export generated reports directly to PDF for sharing purposes.  
        Output: 1  

     3. Multi-language support. Please add support for additional languages such as Spanish and French to accommodate a wider user base.  
        Output: 1  

     4. Two-factor authentication. Introducing two-factor authentication would greatly enhance account security for users.  
        Output: 1  

     5. Drag and drop functionality for file uploads. Users should be able to drag and drop files instead of browsing manually.  
        Output: 1  

3. **Question (Label: `2`)**  
   - **Definition:**  
     Inquiries seeking information, clarification, or troubleshooting assistance about an application’s functionality, configuration, or usage.  
   - **Examples:**  
     1. How to set up database connection? I'm trying to connect to MySQL, but I keep getting a connection timeout error. What settings should I check?  
        Output: 2  

     2. Is there a way to increase API rate limits? I'm facing issues with API limits and need guidance on how to increase them for my use case.  
        Output: 2  

     3. Why is my query running slow? A simple query is taking a long time to execute, even with indexed columns. Any optimization suggestions?  
        Output: 2  

     4. Steps to integrate with third-party service. I want to integrate with Stripe API but don't know where to start. Any guidance?  
        Output: 2  

     5. Difference between free and premium plans. Can you explain the differences in features between the free and premium versions of the product?  
        Output: 2  

4. **Documentation Issue (Label: `3`)**  
   - **Definition:**  
     Reports related to gaps, inaccuracies, or ambiguities in user manuals, API references, tutorials, guides, or other instructional content.  
   - **Examples:**  
     1. Missing steps in installation guide. The installation guide does not mention how to configure environment variables, which caused confusion.  
        Output: 3  

     2. API documentation lacks examples. The API reference does not provide enough code examples to help new users understand usage patterns.  
        Output: 3  

     3. Incorrect parameter details in user guide. The user guide lists incorrect parameter names for the `getData` function.  
        Output: 3  

     4. Update outdated screenshots. Several screenshots in the user manual are outdated and do not match the current UI.  
        Output: 3  

     5. Clarify permission requirements. Documentation should clarify the required user roles to access certain features in the app.  
        Output: 3  

Classification Task Instructions:

1. Analyze the provided issue description.
2. Classify it into one of the four categories:
   - **Bug (`0`)**
   - **Feature Request (`1`)**
   - **Question (`2`)**
   - **Documentation (`3`)**
3. Provide only the numerical label as the output, with no additional text.

"""
]