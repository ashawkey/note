# Google protocol buffers

protobuf is developed to **serialize and retrieve structured data** with high efficiency.

### Basics

```protobuf
// addressbook.proto
syntax = "proto3";

package tutorial;

message Person {
  required string name = 1;
  required int32 id = 2;
  optional string email = 3;

  enum PhoneType {
    MOBILE = 0;
    HOME = 1;
    WORK = 2;
  }

  message PhoneNumber {
    required string number = 1;
    optional PhoneType type = 2 [default = HOME];
  }

  repeated PhoneNumber phones = 4;
}

message AddressBook {
  repeated Person people = 1;
}
```

* message

  like a class, contains fields and subclasses.

* field modifiers:

  * required: must be provided.
  * optional: if not set, use the default, if default is not set, use the system default.
  * repeated: this field may be repeated for 0 to any times. (like dynamic arrays)

* compile

  `protoc -I=$src_dir --python_out=$src_dir $src_dir/addressook.proto`

  >  `-I` is short for `--proto_path`

  `protoc --python_out=./ *.proto`

  this generate `*_pb2.py` files in the python_out directory.



* API

  ```python
  import addressbook_pb2
  person = addressbook_pb2.Person()
  person.name = "John Doe" # req
  person.id = 1234 # req
  person.email = "jdoe@example.com" # opt
  
  phone = person.phones.add() # add a repeated field
  phone.number = "555-4321"
  phone.type = addressbook_pb2.Person.HOME # or simply `phone.type = 1`
  ```



### Proto3

##### Message

can be viewed as class in python.

##### Field

* field type

  * bool, string, double, float, int32, int64, uint32, uint64
  * sint32, sint64 (signed int, more efficient in coding negative integers than regular int32 and int64)
  * enumeration
  * other self-defined message type

* field number

  each field has a unique number in the message.

  best in [1, 15] for only one byte to encode. (Not use 0)

  > the exact range is $[1, 2^{29}-1]$ \ [19000, 19999] 

* field modifier

  * singular (by default)

    proto3 removed required, so there is only optional singulars.

  * repeated

* enum types

  the first enum should start from 0. 

  > exact range is uint32

  ```protobuf
  enum Corpus {
      UNIVERSAL = 0; // which is the default value too.
      WEB = 1;
  }
  
  enum EnumAllowingAlias {
    option allow_alias = true; // allow same enum for a number.
    UNKNOWN = 0;
    STARTED = 1;
    RUNNING = 1;
  }
  enum EnumNotAllowingAlias {
    UNKNOWN = 0;
    STARTED = 1;
    // RUNNING = 1;  // Uncommenting this line will cause a compile error inside Google and a warning message outside.
  }
  ```




##### comments

same as C/C++.



##### reserved field

```protobuf
message Foo {
  reserved 2, 15, 9 to 11;
  reserved "foo", "bar";
}
```

you should reserve the field number or name for your deleted fields, in case of later modifications reuse these fields and cause bugs.



##### import

```protobuf
// old.proto
import public "new.proto"
import "other.proto";
```

when `old.proto` is imported by another proto file saying `client.proto,` 

`client.proto` also import `new.proto`, but doesn't import `other.proto`.

​	

##### package

python will ignore package declarations, since python modules are organized according to locations in file system.



##### oneof

oneof is a type that means the message will only record one of its fields or subclasses.

```protobuf
message SampleMessage {
  oneof test_oneof {
    string name = 4;
    SubMessage sub_message = 9;
  }
}
```

```java
SampleMessage message;
message.set_name("name");
CHECK(message.has_name());
message.mutable_sub_message();   // Will clear name field.
CHECK(!message.has_name());
```



