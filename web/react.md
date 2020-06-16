# React

### JSX

Use </> in code. Place expressions inside {}.

```javascript
const name = 'Josh Perez';

// JSX
const element = (
    <h1 className='test'>
    	Hello, {name}.
    </h1>
);

// equals to
const element = React.createElement(
  'h1',
  {className: 'test'},
  'Hello, ' + name + '.'
);
```



### Render

```javascript
// React Elements are constant.
const element = <h1> Hello World! </h1>;
ReactDOM.render(element, document.getElementById('root'));

// Update
function tick() {
  const element = (
    <div>
      <h1>Hello, world!</h1>
      <h2>It is {new Date().toLocaleTimeString()}.</h2>
    </div>
  );
  ReactDOM.render(element, document.getElementById('root'));
}

setInterval(tick, 1000);
```



### Component & Prop[ertie]s

```javascript
// Function
function welcome(props){
    return <h1>Hello, {props.name}</h1>;
}

// Class (ES6)
class Welcome extends React.Component {
    render() {
        return <h1>Hello, {this.props.name}</h1>;
    }
}
```

* Create Element through Component.

  ```javascript
  const element = <Welcome name='Sara' />; // this.props.name = 'Sara'
  ```

  Class Component name Must start with a Capital Letter.

  **`<div/>` is a HTML tag, while `<Div />` may be  a component.**

* Compose Components

  ```javascript
  function Welcome(props) {
    return <h1>Hello, {props.name}</h1>;
  }
  
  function App() {
    return (
      <div>
        <Welcome name="Sara" />
        <Welcome name="Cahal" />
        <Welcome name="Edite" />
      </div>
    );
  }
  
  ReactDOM.render(
    <App />,
    document.getElementById('root')
  );
  ```

* **Props are Read-Only**



### States & Lifecycle Methods

We use class component to implement dynamic states.

```javascript
class Clock extends React.Component {
  constructor(props) {
    super(props);
    this.state = {date: new Date()};
  }
  // Mount, run on render
  componentDidMount() {
    this.timerID = setInterval(
      () => this.tick(),
      1000
    );
  }
  // Unmount, run on destruct
  componentWillUnmount() {
    clearInterval(this.timerID);
  }
  // setState: Core of dynamics.
  tick() {
    this.setState({
      date: new Date()
    });
  }
  render() {
    return (
      <div>
        <h1>Hello, world!</h1>
        <h2>It is {this.state.date.toLocaleTimeString()}.</h2>
      </div>
    );
  }
}
```

Only use `setState()` to update state. 

* **`setState()` is Asynchronous!**

  ```javascript
  incrementCount() {
    // 注意：这样 *不会* 像预期的那样工作。
    this.setState({count: this.state.count + 1});
  }
  
  handleSomething() {
    // 假设 `this.state.count` 从 0 开始。
    this.incrementCount();
    this.incrementCount();
    this.incrementCount();
    // 当 React 重新渲染该组件时，`this.state.count` 会变为 1，而不是你期望的 3。
  
    // 这是因为上面的 `incrementCount()` 函数是从 `this.state.count` 中读取数据的，
    // 但是 React 不会更新 `this.state.count`，直到该组件被重新渲染。
    // 所以最终 `incrementCount()` 每次读取 `this.state.count` 的值都是 0，并将它设为 1。
  
    // 问题的修复参见下面的说明。
  }
  ```

  Instead, pass in a function to `setState()` solves this problem.

  ```javascript
  incrementCount() {
    this.setState((state) => {return {count: state.count + 1}});
    // 重要：在更新的时候读取 `state`，而不是 `this.state`。
  }
  
  handleSomething() {
    // 假设 `this.state.count` 从 0 开始。
    this.incrementCount();
    this.incrementCount();
    this.incrementCount();
  
    // 如果你现在在这里读取 `this.state.count`，它还是会为 0。
    // 但是，当 React 重新渲染该组件时，它会变为 3。
  }
  ```

  

### Event

* Click

  ```javascript
  class Toggle extends React.Component {
    constructor(props) {
      super(props);
      this.state = {isToggleOn: true};
  
      // 为了在回调中使用 `this`，这个绑定是必不可少的
      this.handleClick = this.handleClick.bind(this);
    }
  
    handleClick() {
      this.setState(state => ({
        isToggleOn: !state.isToggleOn
      }));
    }
  
    render() {
      return (
        <button onClick={this.handleClick}>
          {this.state.isToggleOn ? 'ON' : 'OFF'}
        </button>
      );
    }
  }
  
  ReactDOM.render(
    <Toggle />,
    document.getElementById('root')
  );
  ```

  

### Conditional Rendering

* Avoid render

  Return null to avoid rendering.

  ```javascript
  function WarningBanner(props) {
    if (!props.warn) {
      return null;
    }
  
    return (
      <div className="warning">
        Warning!
      </div>
    );
  }
  
  class Page extends React.Component {
    constructor(props) {
      super(props);
      this.state = {showWarning: true};
      this.handleToggleClick = this.handleToggleClick.bind(this);
    }
  
    handleToggleClick() {
      this.setState(state => ({
        showWarning: !state.showWarning
      }));
    }
  
    render() {
      return (
        <div>
          <WarningBanner warn={this.state.showWarning} />
          <button onClick={this.handleToggleClick}>
            {this.state.showWarning ? 'Hide' : 'Show'}
          </button>
        </div>
      );
    }
  }
  
  ReactDOM.render(
    <Page />,
    document.getElementById('root')
  );
  ```

  

### List & Key

```javascript
function NumberList(props) {
  const numbers = props.numbers;
  const listItems = numbers.map((number) =>
    <li key={number.toString()}> // should provide a unique key.
      {number}
    </li>
  );
  return (
    <ul>{listItems}</ul>
  );
}

const numbers = [1, 2, 3, 4, 5];
ReactDOM.render(
  <NumberList numbers={numbers} />,
  document.getElementById('root')
);
```



### Form

* `setState()` 

  ```javascript
  // ===== input =====
  class NameForm extends React.Component {
    constructor(props) {
      super(props);
      this.state = {value: ''};
  
      this.handleChange = this.handleChange.bind(this); // must
      this.handleSubmit = this.handleSubmit.bind(this); // must
    }
  
    handleChange(event) {
      this.setState({value: event.target.value});
    }
  
    handleSubmit(event) {
      alert('提交的名字: ' + this.state.value);
      event.preventDefault(); // prevent default behavior.
    }
  
    render() {
      return (
        <form onSubmit={this.handleSubmit}>
          <label>
            名字:
            <input type="text" value={this.state.value} onChange={this.handleChange} />
          </label>
          <input type="submit" value="提交" />
        </form>
      );
    }
  }
  
  //===== select =====
  
  class FlavorForm extends React.Component {
    constructor(props) {
      super(props);
      this.state = {value: 'coconut'};
  
      this.handleChange = this.handleChange.bind(this);
      this.handleSubmit = this.handleSubmit.bind(this);
    }
  
    handleChange(event) {
      this.setState({value: event.target.value});
    }
  
    handleSubmit(event) {
      alert('你喜欢的风味是: ' + this.state.value);
      event.preventDefault();
    }
  
    render() {
      return (
        <form onSubmit={this.handleSubmit}>
          <label>
            选择你喜欢的风味:
            <select value={this.state.value} onChange={this.handleChange}>
              <option value="grapefruit">葡萄柚</option>
              <option value="lime">酸橙</option>
              <option value="coconut">椰子</option>
              <option value="mango">芒果</option>
            </select>
          </label>
          <input type="submit" value="提交" />
        </form>
      );
    }
  }
  ```

* `Formik`



### 





