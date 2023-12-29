# React

### start

```bash
npx create-react-app <name>
```


### JSX

Use </> in code. Place expressions inside {}.

```javascript
const name = 'Josh Perez';

// JSX
const element = (
    <h1 className='test'>
    {/* JSX comments */}
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

* extend Attributes 

  ```javascript
  function App1() {
    return <Greeting firstName="Ben" lastName="Hector" />;
  }
  
  function App2() {
    const props = {firstName: 'Ben', lastName: 'Hector'};
    return <Greeting {...props} />;
  }
  ```

* Some equal expressions

  ```javascript
  <MyComponent message="hello world" />
  <MyComponent message={'hello world'} /> // expression returns the string
      
  <MyComponent message="&lt;3" />
  <MyComponent message={'<3'} />
      
  <MyTextBox autocomplete /> // default = true
  <MyTextBox autocomplete={true} />
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

* Default Behavior

  eg. click `<a>` will navigate to `href`.

  eg. click a `submit` input in a form will submit to server.

  eg. click and drag on `textarea` will select.

  How to prevent:

  ```javascript
  event.preventDefault();
  ```

  

* `<button onClick={func}> `

  ```javascript
  class Toggle extends React.Component {
    constructor(props) {
      super(props);
      this.state = {isToggleOn: true};
      this.handleClick = this.handleClick.bind(this); // necessary
    }
  
    handleClick(event) {
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
  
* `event`

  ```javascript
  event.type // "click"
  event.target // 
  ```

  

### Conditional Rendering

* Logic Expressions

  ```javascript
  return (
    <div>
      {warn === true && <p> Warning! </p>}
    </div>
  );
  ```

  

* Return null to avoid rendering.

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


### Dynamic Rendering

```javascript
function DynamicTable() {
  const elements = ['one', 'two', 'three'];
  return (
    <ul>
      {elements.map((value, index) => {
        return <li key={index}>{value}</li>
      })}
    </ul>
  )
}

function DynamicForm(){
  const [inputFields, setInputFields] = useState([{ firstName: '', lastName: '' }]);
  const handleAddFields = () => {
    const values = [...inputFields];
    values.push({ firstName: '', lastName: '' });
    setInputFields(values);
  };

  const handleRemoveFields = index => {
    const values = [...inputFields];
    values.splice(index, 1);
    setInputFields(values);
  };
    
  return (
    <>
      <form onSubmit={handleSubmit}>
        <div className="form-row">
          {inputFields.map((inputField, index) => (
            <Fragment key={index}>
              <div className="form-group col-sm-6">
                <label htmlFor="firstName">First Name</label>
                <input type="text" id="firstName" name="firstName" value={inputField.firstName} />
              </div>
              <div className="form-group col-sm-4">
                <label htmlFor="lastName">Last Name</label>
                <input type="text" id="lastName" name="lastName" value={inputField.lastName} />
              </div>
              <div className="form-group col-sm-2">
                <button className="btn btn-link" type="button"> - </button>
                <button className="btn btn-link" type="button"> + </button>
              </div>
            </Fragment>
          ))}
        </div>
        <div className="submit-button">
          <button className="btn btn-primary mr-2" type="submit" onSubmit={handleSubmit}>
            Save
          </button>
        </div>
      </form>
    </>
  )
}
```


### Refs

To access Child node in Parent node.

* Create

  ```javascript
  class MyComponent extends React.Component {
    constructor(props) {
      super(props);
      this.myRef = React.createRef();
    }
    render() {
      return <div ref={this.myRef} />;
    }
  }
  ```

* Access

  ```javascript
  const node = this.myRef.current;
  ```

* Usage

  We cannot use Ref on Functional Components, since they have no instance.

  ```javascript
  class CustomTextInput extends React.Component {
    constructor(props) {
      super(props);
      // 创建一个 ref 来存储 textInput 的 DOM 元素
      this.textInput = React.createRef();
      this.focusTextInput = this.focusTextInput.bind(this);
    }
  
    focusTextInput() {
      // 直接使用原生 API 使 text 输入框获得焦点
      // 注意：我们通过 "current" 来访问 DOM 节点
      this.textInput.current.focus();
    }
  
    render() {
      // 告诉 React 我们想把 <input> ref 关联到
      // 构造器里创建的 `textInput` 上
      return (
        <div>
          <input
            type="text"
            ref={this.textInput} />
          <input
            type="button"
            value="Focus the text input"
            onClick={this.focusTextInput}
          />
        </div>
      );
    }
  }
  ```


### Context

provide Global `props`  (themes, login-status) used by many components.

```javascript
// Context 可以让我们无须明确地传遍每一个组件，就能将值深入传递进组件树。
// 为当前的 theme 创建一个 context（“light”为默认值）。
const ThemeContext = React.createContext('light');

class App extends React.Component {
  render() {
    // 使用一个 Provider 来将当前的 theme 传递给以下的组件树。
    // 无论多深，任何组件都能读取这个值。
    // 在这个例子中，我们将 “dark” 作为当前的值传递下去。
    return (
      <ThemeContext.Provider value="dark">
        <Toolbar />
      </ThemeContext.Provider>
    );
  }
}

// 中间的组件再也不必指明往下传递 theme 了。
function Toolbar() {
  return (
    <div>
      <ThemedButton />
    </div>
  );
}

class ThemedButton extends React.Component {
  // 指定 contextType 读取当前的 theme context。
  // React 会往上找到最近的 theme Provider，然后使用它的值。
  // 在这个例子中，当前的 theme 值为 “dark”。
  static contextType = ThemeContext; // either use it here. 
  render() {
    return <Button theme={this.context} />;
  }
}

//ThemedButton.contextType = ThemeContext; // or use it here.
```

* `const myContext = React.createContext(defaultValue);`

* `<myContext.Provider value={}> ... </myContext.Provider>`

  Components inside <Provider> will find the nearest context.

* `<myContext.Consumer> {value => ()} </myContext.Consumer>`

  For Functional Components without `this.context`, use this to get `value`.

  

```javascript
// Multiple Context
// Theme context，默认的 theme 是 “light” 值
const ThemeContext = React.createContext('light');

// 用户登录 context
const UserContext = React.createContext({
  name: 'Guest',
});

class App extends React.Component {
  render() {
    const {signedInUser, theme} = this.props;

    // 提供初始 context 值的 App 组件
    return (
      <ThemeContext.Provider value={theme}>
        <UserContext.Provider value={signedInUser}>
          <Layout />
        </UserContext.Provider>
      </ThemeContext.Provider>
    );
  }
}

function Layout() {
  return (
    <div>
      <Sidebar />
      <Content />
    </div>
  );
}

// 一个组件可能会消费多个 context
function Content() {
  return (
    <ThemeContext.Consumer>
      {theme => (
        <UserContext.Consumer>
          {user => (
            <ProfilePage user={user} theme={theme} />
          )}
        </UserContext.Consumer>
      )}
    </ThemeContext.Consumer>
  );
}

```


### Render Props

pass in a `render` attributes to dynamically render component inside component.

```javascript
class Cat extends React.Component {
  render() {
    const mouse = this.props.mouse;
    return (
      <img src="/cat.jpg" style={{ position: 'absolute', left: mouse.x, top: mouse.y }} />
    );
  }
}

class Mouse extends React.Component {
  constructor(props) {
    super(props);
    this.handleMouseMove = this.handleMouseMove.bind(this);
    this.state = { x: 0, y: 0 };
  }

  handleMouseMove(event) {
    this.setState({
      x: event.clientX,
      y: event.clientY
    });
  }

  render() {
    return (
      <div style={{ height: '100vh' }} onMouseMove={this.handleMouseMove}>
        /*
          使用 `render`prop 动态决定要渲染的内容，
          而不是给出一个 <Mouse> 渲染结果的静态表示
        */
        {this.props.render(this.state)}
      </div>
    );
  }
}

class MouseTracker extends React.Component {
  render() {
    return (
      <div>
        <h1>移动鼠标!</h1>
        <Mouse render={mouse => (<Cat mouse={mouse} />)}/>  // render prop
      </div>
    );
  }
}
```


### Hook

It allows **functional components** to use `states`. (Avoid annoying `this` in class components!)

Cannot be used in **class components**.

* `useState(state)`

  ```javascript
  import React, { useState } from 'react';
  
  function Example() {
    // 声明一个叫 “count” 的 state 变量。
    const [count, setCount] = useState(0);
  
    return (
      <div>
        <p>You clicked {count} times</p>
        <button onClick={() => setCount(count + 1)}>
          Click me
        </button>
      </div>
    );
  }
  
  // async timer example
  function AsyncExample() {
  
    const [count, setCount] = useState(0);
  
    return (
      <div>
        <p>You clicked {count} times</p>
        <button onClick={() => setCount(count + 1)}>
          Click me
        </button>
      </div>
    );
  }
  ```

  

* `useEffect(func, [condition])`

  ```javascript
  import React, { useState, useEffect } from 'react';
  
  function FriendStatus(props) {
    const [isOnline, setIsOnline] = useState(null);
  
    useEffect(() => {
      function handleStatusChange(status) {
        setIsOnline(status.isOnline);
      }
      // replace componentDidMount & componentDidUpdate:
      ChatAPI.subscribeToFriendStatus(props.friend.id, handleStatusChange);
      // replace componentWillUnmount:
      return function cleanup() {
        ChatAPI.unsubscribeFromFriendStatus(props.friend.id, handleStatusChange);
      };
    });
  
    if (isOnline === null) {
      return 'Loading...';
    }
    return isOnline ? 'Online' : 'Offline';
  }
  ```

  this equals to:
  
  ```javascript
  class FriendStatus extends React.Component {
    constructor(props) {
      super(props);
      this.state = { isOnline: null };
      this.handleStatusChange = this.handleStatusChange.bind(this);
    }
  
    componentDidMount() {
      ChatAPI.subscribeToFriendStatus(
        this.props.friend.id,
        this.handleStatusChange
      );
    }
    componentWillUnmount() {
      ChatAPI.unsubscribeFromFriendStatus(
        this.props.friend.id,
        this.handleStatusChange
      );
    }
    handleStatusChange(status) {
      this.setState({
        isOnline: status.isOnline
      });
    }
  
  render() {
      if (this.state.isOnline === null) {
        return 'Loading...';
      }
      return this.state.isOnline ? 'Online' : 'Offline';
    }
  }
  ```
  
  Conditional Effect:
  
  ```javascript
  useEffect(
    () => {
      const subscription = props.source.subscribe();
      return () => {
        subscription.unsubscribe();
      };
    },
    [props.source], // called only if props.source changed.
  );
  
  useEffect(
    () => {
    const subscription = props.source.subscribe();
      return () => {
      subscription.unsubscribe();
      };
    },
    [], // Never Update.
  );
  ```
  
  
  
* `useContext(myContext)`

  Also need <Provider>.

  ```javascript
  const themes = {
    light: {
      foreground: "#000000",
      background: "#eeeeee"
    },
    dark: {
      foreground: "#ffffff",
      background: "#222222"
    }
  };
  
  const ThemeContext = React.createContext(themes.light);
  
  function App() {
    return (
      <ThemeContext.Provider value={themes.dark}>
        <Toolbar />
      </ThemeContext.Provider>
    );
  }
  
  function Toolbar(props) {
    return (
      <div>
        <ThemedButton />
      </div>
    );
  }
  
  function ThemedButton() {
    const theme = useContext(ThemeContext); // replace <Consumer> or this.context
    return (
      <button style={{ background: theme.background, color: theme.foreground }}>
        I am styled by theme context!
      </button>
    );
  }
  ```

  

  

  

### Fragments

```javascript
class Table extends React.Component {
  render() {
    return (
      <table>
        <tr>
          <Columns />
        </tr>
      </table>
    );
  }
}

class Columns extends React.Component {
  render() {
    // wrong! <div> ruins <table>
    return (
      <div>
        <td>Hello</td>
        <td>World</td>
      </div>
    );
  }
}
```

Instead, use:

```javascript
class Columns extends React.Component {
  render() {
    return (
      <React.Fragment>
        <td>Hello</td>
        <td>World</td>
      </React.Fragment>
    );
  }
}

// or simply:
class Columns extends React.Component {
  render() {
    return (
      <>
        <td>Hello</td>
        <td>World</td>
      </>
    );
  }
}
```


### Other API

* `React.PureComponent`

  It has `shouldComponentUpdate()` method, allows shallow comparison to decide time to update.

  It improves performance only when `props & states` are simple.

* `React.memo`

  function component wrapper to improve performance.

  Similar to `PureComponent`.

* `React.createElement(type[, props, children])`

  JSX will be converted to this function by Babel.

* `React.cloneElement(element[, props, children])`

  Clone an element with props shallow-merged.

  

  


