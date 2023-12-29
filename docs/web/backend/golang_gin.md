# Gin

A Web Framework in Go. (https://godoc.org/github.com/gin-gonic/gin)

### Overview

```go
package main

import "github.com/gin-gonic/gin"

func pong(c *gin.Context) {
    // JSON(): serialize and send return code and object to JSON.
    // H: short for map[string]interface{}
    c.JSON(200, gin.H{
        "message": "pong",
    })
}

func main() {
    r := gin.Default()
    r.GET("/ping", pong) // "/ping" is routed to pong()
	r.Run() // listen and serve on 0.0.0.0:8080
}
```

### Route

* GET

  ```javascript
  // with URL parameters
  fetch('/api?action=getcomment'+'&pid='+pid).then(...)
  ```

  ```go
  r.GET("/api", apiGet)
  
  func apiGet(c *gin.Context) {
  	action := c.Query("action")
  	switch action {
  	case "getcomment":
  		getOne(c)
  		return
  	case "search":
  		searchPost(c)
  		return
  	default:
  		c.AbortWithStatus(403)
  	}
  }
  
  func searchPost(c *gin.Context) {
  	page, err := strconv.Atoi(c.Query("page"))
  	pageSize, err := strconv.Atoi(c.Query("pagesize"))
  	keywords := c.Query("keywords")
  
  	data, err2 := dbSearchSavedPosts(strings.ReplaceAll(keywords, " ", " +"), (page-1)*pageSize, pageSize)
  	if err2 != nil {
  		log.Printf("dbSearchSavedPosts failed while searchList: %s\n", err2)
  		httpReturnWithCodeOne(c, "数据库读取失败，请联系管理员")
  		return
  	} else {
  		c.JSON(http.StatusOK, gin.H{
  			"code":      0,
  			"data":      IfThenElse(data != nil, data, []string{}),
  			"timestamp": getTimeStamp(),
  			"count":     IfThenElse(data != nil, len(data), 0),
  		})
  		return
  	}
  }
  ```

* POST

  * URL parameters

    ```javascript
    fetch(
        'api/login'
        +'?user='+encodeURIComponent(this.username_ref.current.value)
        +'&valid_code='+encodeURIComponent(this.password_ref.current.value), 
        {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                excluded_scopes: this.state.excluded_scopes||[],
            }),
        }
    ).then(...)
    ```

    Use `c.Query`.

    ```go
    r.POST('/api/login', login)
    
    func login(c *gin.Context) {
    	user := c.Query("user")
    	code := c.Query("valid_code")
        
    	hashedUser := hashEmail(user)
    	token := genToken()
        
    	err = dbSaveToken(token, hashedUser)
    	if err != nil {
    		c.JSON(http.StatusOK, gin.H{
    			"code": 1,
    			"msg": "数据库写入失败，请联系管理员",
    		})
    		return
    	} else {
    		c.JSON(http.StatusOK, gin.H{
    			"code": 0,
    			"msg": "登录成功！",
    			"user_token": token,
    		})
    		return
    	}
    }
    ```

  * Form

    Use `c.PostForm`.

    ```go
    r := gin.Default()
    r.POST("/", func(c *gin.Context) {
        wechat := c.PostForm("wechat")
        c.String(200, wechat)
    })
    ```

    
### CORS

```go
package main

import (
	"time"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
)

func main() {
	router := gin.Default()
	// CORS for https://foo.com and https://github.com origins, allowing:
	// - PUT and PATCH methods
	// - Origin header
	// - Credentials share
	// - Preflight requests cached for 12 hours
	router.Use(cors.New(cors.Config{
		AllowOrigins:     []string{"https://foo.com"},
		AllowMethods:     []string{"PUT", "PATCH"},
		AllowHeaders:     []string{"Origin"},
		ExposeHeaders:    []string{"Content-Length"},
		AllowCredentials: true,
		AllowOriginFunc: func(origin string) bool {
			return origin == "https://github.com"
		},
		MaxAge: 12 * time.Hour,
	}))
	router.Run()
}
```

