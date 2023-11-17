## datetime

Use datetime to calculate date and time!

```python
import datetime

### simple delta calender
a = datetime.datetime(year=2024, month=5, day=20) # [year, month, day](required), hour, minute, second
d = datetime.timedelta(weeks=25) # seconds, minute, hours, days, weeks

b = a + d
print(b) # 2024-11-11
print(b.weekday()) # 0, so Monday

### today
t = datetime.datetime.today()
```

