### Color

* Why multiply mixes color:

  ```cpp
  glm::vec3 lightColor(1.0f, 1.0f, 1.0f); // white light
  glm::vec3 toyColor(1.0f, 0.5f, 0.31f); // defines the reflections
  glm::vec3 result = lightColor * toyColor; // = (1.0f, 0.5f, 0.31f); Red is totally reflected, Green is 50% reflected, ...
  
  glm::vec3 lightColor(0.0f, 1.0f, 0.0f); // green light
  glm::vec3 toyColor(1.0f, 0.5f, 0.31f);
  glm::vec3 result = lightColor * toyColor; // = (0.0f, 0.5f, 0.0f); No Red & Blue light to reflect.
  ```

* Let there be light!

  Create a light VAO:

  ```cpp
  unsigned int lightVAO;
  glGenVertexArrays(1, &lightVAO);
  glBindVertexArray(lightVAO);
  // 只需要绑定VBO不用再次设置VBO的数据，因为箱子的VBO数据中已经包含了正确的立方体顶点数据
  glBindBuffer(GL_ARRAY_BUFFER, VBO);
  // 设置灯立方体的顶点属性（对我们的灯来说仅仅只有位置数据）
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
  glEnableVertexAttribArray(0);
  ```

  


### gamma correction

