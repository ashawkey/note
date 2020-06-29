# Form

### Controlled Form

```javascript
function NameForm() {
  // hook
  const [name, setName] = useState('');
  // event handler
  function handleSubmit(event) {
    alert('Submitted Name: ' + name);
    event.preventDefault();
  }
  function handleChange(event) {
    setName(event.target.value); // here event.target is <input>
  }
  // Controlled form, onSubmit will call handleSubmit(event)
  return (
    <form onSubmit={handleSubmit}> 
      // input, onChange will call handleChange(event)
      <label> Name: <input type="text" value={name} onChange={handleChange}/> </label>
      <input type="submit" value="提交" />
    </form>
  );
}

function SelectForm() {
  const [value, setValue] = useState('coconut');
  ...
  return (
    <form onSubmit={handleSubmit}>
      <label>
        选择你喜欢的风味:
          <select value={value} onChange={handleChange}>
            <option value="grapefruit"> 葡萄柚 </option>
            <option value="lime">       酸橙   </option>
            <option value="coconut">    椰子   </option>
            <option value="mango">      芒果   </option>
          </select>
        </label>
        <input type="submit" value="提交" />
    </form>
  );
}
```



### Uncontrolled Form: File Input 

```javascript
class FileInput extends React.Component {
  constructor(props) {
    super(props);
    this.handleSubmit = this.handleSubmit.bind(this);
    this.fileInput = React.createRef();
  }
  handleSubmit(event) {
    event.preventDefault();
    alert(
      `Selected file - ${this.fileInput.current.files[0].name}`
    );
  }

  render() {
    return (
      <form onSubmit={this.handleSubmit}>
        <label>
          Upload file:
          <input type="file" ref={this.fileInput} />
        </label>
        <br />
        <button type="submit">Submit</button>
      </form>
    );
  }
}

ReactDOM.render(
  <FileInput />,
  document.getElementById('root')
);
```





### Formik & Yup

```javascript
import React from 'react';
import { Formik, Field, Form, ErrorMessage } from 'formik';
import * as Yup from 'yup';

const SignupForm = () => {
  return (
    <Formik
      initialValues={{ firstName: '', lastName: '', email: '' }}
      validationSchema={Yup.object({
        firstName: Yup.string()
          .max(15, 'Must be 15 characters or less')
          .required('Required'),
        lastName: Yup.string()
          .max(20, 'Must be 20 characters or less')
          .required('Required'),
        email: Yup.string()
          .email('Invalid email address')
          .required('Required'),
      })}
      onSubmit={(values, { setSubmitting }) => {
        // run the function after 400 ms
        setTimeout(() => {
          alert(JSON.stringify(values, null, 2));
          setSubmitting(false);
        }, 400);
      }}
    >
      <Form>
        <label htmlFor="firstName">First Name</label>
        <Field name="firstName" type="text" />
        <ErrorMessage name="firstName" />
        <label htmlFor="lastName">Last Name</label>
        <Field name="lastName" type="text" />
        <ErrorMessage name="lastName" />
        <label htmlFor="email">Email Address</label>
        <Field name="email" type="email" />
        <ErrorMessage name="email" />
        <button type="submit">Submit</button>
      </Form>
    </Formik>
  );
};
```



### Dynamic Form

```javascript

```

