# safetensors-java

## Python -> Java

```python
#!/usr/bin/env python

from safetensors.torch import save_file

import torch

tensors = {
    "some_ints": torch.tensor([[-1, 0, 1, 2]]),
    "some_floats": torch.tensor([[[-1.0, 0.0], [1.0, 2.0]]]),
}

save_file(tensors, "sample.safetensors")
```

```java
Safetensors sample = SafetensorsViewer.load(new File("sample.safetensors"));
{
    String tensorName = "some_ints";
    {
        List<Integer> shape = sample.getHeader().get(tensorName).getShape();
        assert shape.equals(Arrays.asList(1, 4));
    }
    {
        LongBuffer longBuffer = sample.getLongBuffer(tensorName);
        long[] longs = new long[longBuffer.limit()];
        longBuffer.get(longs);
        assert Arrays.equals(longs, new long[] {-1L, 0L, 1L, 2L});
    }
}
{
    String tensorName = "some_floats";
    {
        List<Integer> shape = sample.getHeader().get(tensorName).getShape();
        assert shape.equals(Arrays.asList(1, 2, 2));
    }
    {
        FloatBuffer floatBuffer = sample.getFloatBuffer(tensorName);
        float[] floats = new float[floatBuffer.limit()];
        floatBuffer.get(floats);
        assert Arrays.equals(floats, new float[] {-1.0f, 0.0f, 1.0f, 2.0f});
    }
}
```

## Java -> Python

```java
{
    SafetensorsBuilder safetensorsBuilder = new SafetensorsBuilder();
    {
        List<Integer> shape = Arrays.asList(1, 4);
        long[] longs = new long[]{-1L, 0L, 1L, 2L};
        safetensorsBuilder.add("some_ints", shape, longs);
    }
    {
        List<Integer> shape = Arrays.asList(1, 2, 2);
        float[] floats = new float[]{-1.0f, 0.0f, 1.0f, 2.0f};
        safetensorsBuilder.add("some_floats", shape, floats);
    }
    Safetensors subject = safetensorsBuilder.build();
    subject.save(new File("subject.safetensors"));
}
```

```python
#!/usr/bin/env python

from safetensors.torch import load_file

import torch

tensors = load_file("subject.safetensors")
assert torch.equal(tensors["some_ints"], torch.tensor([[-1, 0, 1, 2]]))
assert torch.equal(tensors["some_floats"], torch.tensor([[[-1.0, 0.0], [1.0, 2.0]]]))
```
