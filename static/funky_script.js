// State object to store responses
const state = {
    responses: Array(31).fill(''),
};

// Function to handle input change
const handleChange = (index, value) => {
    state.responses[index] = value;
    console.log(`Response for Question ${index + 1}: ${value}`);
};

// Function to handle form submission
const handleSubmit = () => {
    console.log('All responses:', state.responses);
    // Perform any additional logic for form submission here
    window.alert('Your Predicted Grade is: F');
};

const generateQuestionLabel = (index) => `Question ${index + 1}`;

const questions = [
    "Student Age (1: 18-21, 2: 22-25, 3: above 26)",
    "Sex (1: female, 2: male)",
    "Graduated high-school type: (1: private, 2: state, 3: other)",
    "Scholarship type: (1: None, 2: 25%, 3: 50%, 4: 75%, 5: Full)",
    "Additional work: (1: Yes, 2: No)",
    "Regular artistic or sports activity: (1: Yes, 2: No)",
    "Do you have a partner: (1: Yes, 2: No)",
    "Total salary if available (1: USD 135-200, 2: USD 201-270, 3: USD 271-340, 4: USD 341-410, 5: above 410)",
    "Transportation to the university: (1: Bus, 2: Private car/taxi, 3: bicycle, 4: Other)",
    "Accommodation type in Cyprus: (1: rental, 2: dormitory, 3: with family, 4: Other)",
    "Mothers’ education: (1: primary school, 2: secondary school, 3: high school, 4: university, 5: MSc., 6: Ph.D.)",
    "Fathers’ education: (1: primary school, 2: secondary school, 3: high school, 4: university, 5: MSc., 6: Ph.D.)",
    "Number of sisters/brothers (if available): (1: 1, 2:, 2, 3: 3, 4: 4, 5: 5 or above)",
    "Parental status: (1: married, 2: divorced, 3: died - one of them or both)",
    "Mothers’ occupation: (1: retired, 2: housewife, 3: government officer, 4: private sector employee, 5: self-employment, 6: other)",
    "Fathers’ occupation: (1: retired, 2: government officer, 3: private sector employee, 4: self-employment, 5: other)",
    "Weekly study hours: (1: None, 2: <5 hours, 3: 6-10 hours, 4: 11-20 hours, 5: more than 20 hours)",
    "Reading frequency (non-scientific books/journals): (1: None, 2: Sometimes, 3: Often)",
    "Reading frequency (scientific books/journals): (1: None, 2: Sometimes, 3: Often)",
    "Attendance to the seminars/conferences related to the department: (1: Yes, 2: No)",
    "Impact of your projects/activities on your success: (1: positive, 2: negative, 3: neutral)",
    "Attendance to classes (1: always, 2: sometimes, 3: never)",
    "Preparation to midterm exams 1: (1: alone, 2: with friends, 3: not applicable)",
    "Preparation to midterm exams 2: (1: closest date to the exam, 2: regularly during the semester, 3: never)",
    "Taking notes in classes: (1: never, 2: sometimes, 3: always)",
    "Listening in classes: (1: never, 2: sometimes, 3: always)",
    "Discussion improves my interest and success in the course: (1: never, 2: sometimes, 3: always)",
    "Flip-classroom: (1: not useful, 2: useful, 3: not applicable)",
    "Cumulative grade point average in the last semester (/4.00): (1: <2.00, 2: 2.00-2.49, 3: 2.50-2.99, 4: 3.00-3.49, 5: above 3.49)",
    "Expected Cumulative grade point average in the graduation (/4.00): (1: <2.00, 2: 2.00-2.49, 3: 2.50-2.99, 4: 3.00-3.49, 5: above 3.49)",
];

// Render function
const render = () => {
    const pageElement = document.createElement('div');
    pageElement.className = 'pageStyle';

    const headerElement = document.createElement('div');
    headerElement.className = 'headerStyle';

    const titleElement = document.createElement('h1');
    titleElement.className = 'titleStyle';
    titleElement.textContent = 'Please input your data to predict your end of term performance!';

    headerElement.appendChild(titleElement);
    pageElement.appendChild(headerElement);

    questions.forEach((question, index) => {
        const inputContainerElement = document.createElement('div');
        inputContainerElement.className = 'inputContainerStyle';

        const labelElement = document.createElement('label');
        labelElement.textContent = question;

        const inputElement = document.createElement('input');
        inputElement.type = 'text';
        inputElement.className = 'inputStyle';
        inputElement.placeholder = `Answer for ${generateQuestionLabel(index)}`;
        inputElement.addEventListener('input', (e) => handleChange(index, e.target.value));

        inputContainerElement.appendChild(labelElement);
        inputContainerElement.appendChild(inputElement);

        pageElement.appendChild(inputContainerElement);
    });

    const buttonElement = document.createElement('button');
    buttonElement.textContent = 'Submit';
    buttonElement.className = 'buttonStyle';
    buttonElement.addEventListener('click', handleSubmit);

    pageElement.appendChild(buttonElement);

    document.body.appendChild(pageElement);
};

// Initial rendering
render();

