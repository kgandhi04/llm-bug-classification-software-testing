# prompt.py

classification_prompts = [

    """
You are an advanced text classification model trained to analyze issue descriptions and classify them into one of the following categories based on their content:

1. Bug (0):
   - Describes software defects, errors, or unexpected behavior in an application.
   - Typically includes:
     - Steps to reproduce the issue.
     - Expected vs actual behavior.
     - Error messages, screenshots, or logs.
   - Example keywords: "unexpected error," "crash," "failed to," "not working as expected."

2. Feature (1):
   - Requests for new functionalities, improvements, or enhancements to an application.
   - Generally includes:
     - A clear description of the proposed feature.
     - Reasons why the feature is needed.
     - Potential benefits and use cases.
   - Example keywords: "add support for," "new feature request," "enhancement," "would like to."

3. Question (2):
   - Inquiries seeking information, clarifications, or troubleshooting advice.
   - These may include:
     - How-to questions related to the system.
     - Clarifications about existing functionalities.
     - Troubleshooting requests.
   - Example keywords: "how do I," "why does," "what is," "can you explain."

4. Documentation (3):
   - Issues related to missing, outdated, or unclear documentation.
   - Generally focuses on:
     - Requests for improving existing documentation.
     - Reporting documentation errors.
     - Suggestions for better explanations.
   - Example keywords: "documentation missing," "update the guide," "instructions not clear."

Examples:

Example 1:
Input:
TITLE BUG Clicking on one of the breadcrumb items causes Files instance to crash
BODY When navigating to a folder using breadcrumbs, the Files app crashes every time I click on an item.
Expected behavior: The folder should open.
Actual behavior: The application crashes.
System: Windows 11, Version 22H2.
Output: 0

Example 2:
Input:
TITLE Add multi-language support for profile names
BODY It would be great if the platform could support different language characters for profile names.
Currently, only English characters are supported, which makes it difficult for international users.
Output: 1

Example 3:
Input:
TITLE How to configure automatic backups in the new version?
BODY In the latest version of the software, the backup configuration interface has changed.
Could you provide steps on how to set up automatic daily backups?
Output: 2

Example 4:
Input:
TITLE Outdated API documentation for v3 endpoints
BODY The current API documentation does not cover the v3 endpoints introduced in the latest update.
Some fields and parameters are not explained, causing confusion among developers.
Output: 3

Instruction:
Analyze the following input and classify it into one of the four categories (0, 1, 2, 3) based on the examples above. Provide only the numerical label as the output.

Input:
[TITLE and BODY of the issue here]
Output:
[Your classification]

Just give the final classification as integer.""",


"""
You are an advanced text classification model trained to analyze issue descriptions and classify them into one of the following categories based on their content:

1. Bug (0):
Description:
A bug refers to a software defect, error, or unintended behavior in the application. These issues negatively impact functionality, performance, or user experience and typically require investigation and resolution by the development team. Bugs can manifest as crashes, incorrect outputs, UI glitches, performance degradation, or unexpected behavior.

Common Elements:

Steps to reproduce: Detailed instructions to recreate the issue.
Expected vs actual behavior: Explanation of what should happen vs what actually occurs.
Error messages/logs: Screenshots or console logs showing error details.
Environment details: Operating system, browser version, device type, app version.
Reproducibility: Frequency of the issue (e.g., always, intermittently, under specific conditions).
Impact: Severity level (e.g., critical, major, minor).
Workarounds: Any temporary fixes the user has tried.
Example Keywords:
"unexpected error," "crash," "not responding," "failed to," "not working as expected," "throws an error," "application freezes," "stuck on loading," "timeout occurred," "incorrect calculation," "page not loading," "broken link," "data not saving," "UI glitch," "performance lag," "404/500 error," "system hang," "permissions issue," "duplicate records," "wrong data displayed," "authentication failure," "slow response," "session expired," "unexpected behavior."

Examples:

TITLE: App crashes when uploading large files
BODY: Every time I try to upload a file larger than 100MB, the app crashes unexpectedly.
Steps to reproduce:

Go to the upload page
Select a file > 100MB
Click Upload
Expected behavior: The file should upload successfully.
Actual behavior: The app freezes and closes.
Environment: Windows 10, Chrome 118.
Output: 0
TITLE: Search function returns irrelevant results
BODY: Searching for specific terms shows results that do not match the entered keywords.
Expected behavior: Only relevant items should appear.
Actual behavior: Unrelated items are displayed.
Output: 0

TITLE: Dark mode not applied to all pages
BODY: After enabling dark mode, some sections of the dashboard remain in light mode.
Output: 0

TITLE: Login page shows 'Invalid token' error randomly
BODY: Users occasionally receive an "Invalid token" error when logging in, even though credentials are correct.
Output: 0

TITLE: Payment page freezes during checkout
BODY: The payment page becomes unresponsive after clicking "Proceed to Payment." No error messages appear.
Output: 0

2. Feature (1):
Description:
A feature request is a suggestion for new functionality, improvements, or enhancements to an existing application. These requests aim to add value, improve user experience, or address gaps in current capabilities.

Common Elements:

Feature description: Clear explanation of the desired functionality.
Use case: Real-world scenario where the feature would be useful.
Business impact: How it benefits users or improves efficiency.
Comparison: Reference to competitors having similar features.
Design suggestions: Preferred UI/UX expectations if applicable.
Priority: Whether it's a must-have or a nice-to-have.
Example Keywords:
"add support for," "new feature request," "enhancement," "would like to see," "improve user experience," "optimize performance," "extend functionality," "can we include," "additional option," "suggest to include," "upgrade feature," "new capability," "enhanced reporting," "better integration," "customization option," "improved layout."

Examples:

TITLE: Add multi-language support for profile names
BODY: Currently, only English characters are supported. Adding multi-language support would allow users from different regions to enter their names correctly.
Output: 1

TITLE: Implement dark mode for reports section
BODY: The reports page lacks dark mode support, which is inconvenient for users working at night.
Output: 1

TITLE: Suggest adding a bulk edit option for tasks
BODY: A bulk edit feature would help users modify multiple tasks at once instead of editing them individually.
Output: 1

TITLE: Request for API endpoint to fetch user preferences
BODY: It would be helpful to have an API endpoint that retrieves user preference settings for better customization.
Output: 1

TITLE: Enhance search functionality with autocomplete
BODY: The current search feature could be improved by adding an autocomplete function to suggest relevant results as the user types.
Output: 1

3. Question (2):
Description:
A question involves inquiries seeking information, clarifications, or troubleshooting advice regarding an application's functionality, configuration, or usage. These may arise from confusion about existing features or uncertainty about best practices.

Common Elements:

Topic: Clear subject of the question.
Context: Description of the situation leading to the question.
Expected information: What the user hopes to understand.
Prior attempts: Steps the user has already tried to solve the issue.
Example Keywords:
"how do I," "can I," "what is," "why does," "explain," "troubleshoot," "need help with," "is it possible to," "how to set up," "steps for," "can you clarify," "what's the best way," "guidance on."

Examples:

TITLE: How do I reset my password?
BODY: I forgot my password and can't find an option to reset it. What should I do?
Output: 2

TITLE: What permissions are needed to access the reports section?
BODY: I can’t view reports and want to know the required user roles.
Output: 2

TITLE: Can I export data to CSV format?
BODY: I need to export my reports, but I don't see an option for CSV download.
Output: 2

TITLE: Why does my dashboard show incorrect data?
BODY: Some of the numbers seem off. Is there a way to troubleshoot this?
Output: 2

TITLE: How to integrate the system with Slack notifications?
BODY: I would like to receive updates via Slack, but I need guidance on setting it up.
Output: 2

4. Documentation (3):
Description:
Documentation issues refer to missing, outdated, or unclear guides, instructions, or reference materials related to the application. These reports highlight the need for improvements in documentation to enhance user understanding.

Common Elements:

Specific documentation reference: Which guide or section is unclear.
Impact: How the missing information affects users.
Suggested improvements: Specific areas that need better explanations.
Example Keywords:
"documentation missing," "update the guide," "instructions not clear," "need more details," "incorrect information," "add examples," "API reference incomplete," "tutorial not available," "update screenshots," "unclear steps."

Examples:

TITLE: Update the API documentation for v2 endpoints
BODY: The current API documentation lacks details about the new authentication method.
Output: 3

TITLE: Clarify installation steps for Linux
BODY: The installation guide for Linux lacks dependencies information.
Output: 3

TITLE: Add examples to query documentation
BODY: The SQL query guide could benefit from real-world usage examples.
Output: 3

Instruction:
Analyze the following input and classify it into one of the four categories (0, 1, 2, 3) based on the examples above. Provide only the numerical label as the output.

Input:
[TITLE and BODY of the issue here]
Output:
[Your classification]

Just give the final classification as integer.

"""
    # Add more prompts if needed...
]
