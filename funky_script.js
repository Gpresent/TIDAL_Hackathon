import React, { useState } from 'react';



const TealBackgroundWithInputs = () => {
    const [responses, setResponses] = useState(Array(31).fill(''));

    const handleChange = (index, value) => {
        const updatedResponses = [...responses];
        updatedResponses[index] = value;
        setResponses(updatedResponses);
        console.log(`Response for Question ${index + 1}: ${value}`);
    };

    const handleSubmit = () => {
        // Handle form submission logic here
        console.log('All responses:', responses);

        // Display an alert with the predicted grade
        window.alert(`Your Predicted Grade is: `);
    };

    const pageStyle = {
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        background: '#fff8dc', // Set the background color to #fff8dc
        minHeight: '100vh', // Ensure the content takes at least the full height of the viewport
    };

    const headerStyle = {
        backgroundColor: 'teal',
        width: '100%',
        padding: '20px',
        textAlign: 'center',
    };

    const titleStyle = {
        color: 'white',
        fontSize: '24px',
        marginBottom: '20px',
    };

    const inputContainerStyle = {
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        marginBottom: '25px',
    };

    const inputStyle = {
        margin: '5px',
        padding: '10px',
        textAlign: 'center',
        width: '300px',
    };

    const buttonStyle = {
        backgroundColor: 'green',
        color: 'white',
        padding: '10px 20px',
        fontSize: '16px',
        cursor: 'pointer',
        border: 'none',
        borderRadius: '5px',
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

    return (
        <div className={"pageStyle"}>
            <div className={"headerStyle"}>
                <h1 style={titleStyle}>
                    Please input your data to predict your end of term performance!
                </h1>
            </div>
            {questions.map((question, index) => (
                <div key={index} className={"inputContainerStyle"}>
                    <label>{question}</label>
                    <input
                        type="text"
                        className={"inputStyle"}
                        placeholder={`Answer for ${generateQuestionLabel(index)}`}
                        onChange={(e) => handleChange(index, e.target.value)}
                    />
                </div>
            ))}
            <button onClick={handleSubmit} style={buttonStyle}>
                Submit
            </button>
        </div>
    );
};

export default TealBackgroundWithInputs;
