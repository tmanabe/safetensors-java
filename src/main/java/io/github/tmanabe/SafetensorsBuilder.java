package io.github.tmanabe;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.AbstractMap;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class SafetensorsBuilder {
    private final Map<String, Safetensors.HeaderValue> header = new HashMap<>();
    private final Map<String, Object> bodies = new HashMap<>();

    private int byteSize = 0;

    private void checkLength(List<Integer> shape, int length) {
        if (shape.isEmpty()) {
            if (1 == length) return;
        } else {
            int expect = 1;
            for (Integer i : shape) {
                expect *= i;
            }
            if (expect == length) return;
        }
        throw new IllegalArgumentException("Shape does not match length: " + shape + "," + length);
    }

    public void add(String tensorName, List<Integer> shape, long[] longs) {
        checkLength(shape, longs.length);
        Map.Entry<Integer, Integer> dataOffsets;
        {
            int begin = byteSize;
            byteSize += Long.BYTES * longs.length;
            int end = byteSize;
            dataOffsets = new AbstractMap.SimpleEntry<>(begin, end);
        }
        Safetensors.HeaderValue headerValue = new Safetensors.HeaderValue("I64", shape, dataOffsets);
        header.put(tensorName, headerValue);
        bodies.put(tensorName, longs);
    }

    public void add(String tensorName, List<Integer> shape, float[] floats) {
        checkLength(shape, floats.length);
        Map.Entry<Integer, Integer> dataOffsets;
        {
            int begin = byteSize;
            byteSize += Float.BYTES * floats.length;
            int end = byteSize;
            dataOffsets = new AbstractMap.SimpleEntry<>(begin, end);
        }
        Safetensors.HeaderValue headerValue = new Safetensors.HeaderValue("F32", shape, dataOffsets);
        header.put(tensorName, headerValue);
        bodies.put(tensorName, floats);
    }

    public Safetensors build() {
        ByteBuffer byteBuffer = ByteBuffer.wrap(new byte[byteSize]);
        for (Map.Entry<String, Safetensors.HeaderValue> entry : header.entrySet()) {
            ByteBuffer bb;
            {
                Map.Entry<Integer, Integer> dataOffsets = entry.getValue().getDataOffsets();
                int begin = dataOffsets.getKey();
                int end = dataOffsets.getValue();
                bb = ByteBuffer.wrap(byteBuffer.array(), begin, end - begin).order(ByteOrder.LITTLE_ENDIAN);
            }

            Object object = bodies.get(entry.getKey());
            if (object instanceof long[]) {
                bb.asLongBuffer().put((long[]) object);
                continue;
            }
            if (object instanceof float[]) {
                bb.asFloatBuffer().put((float[]) object);
                continue;
            }
            throw new IllegalArgumentException("Unsupported type: " + object.getClass().getTypeName());
        }
        return new Safetensors(header, byteBuffer);
    }
}
