package io.github.tmanabe;

import java.io.BufferedOutputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.nio.charset.StandardCharsets;
import java.util.AbstractMap;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class SafetensorsBuilder {
    private static class HeaderValue {
        private final String dtype;
        private final List<Integer> shape;
        private final Map.Entry<Integer, Integer> dataOffsets;

        HeaderValue(String dtype, List<Integer> shape, Map.Entry<Integer, Integer> dataOffsets) {
            this.dtype = dtype;
            this.shape = shape;
            this.dataOffsets = dataOffsets;
        }

        private String serialize() {
            StringBuilder shapeBuilder = new StringBuilder();
            shapeBuilder.append('[');
            if (!shape.isEmpty()) {
                for (int i : shape) {
                    shapeBuilder.append(i);
                    shapeBuilder.append(',');
                }
                shapeBuilder.deleteCharAt(shapeBuilder.length() - 1);
            }
            shapeBuilder.append(']');

            String result = "{'dtype':'%s','shape':%s,'data_offsets':[%d,%d]}".replaceAll("'", "\"");
            return String.format(result, dtype, shapeBuilder, dataOffsets.getKey(), dataOffsets.getValue());
        }
    }

    private final Map<String, HeaderValue> header = new HashMap<>();
    private final Map<String, Object> bodies = new HashMap<>();

    private int byteSize = 0;

    private static void checkLength(List<Integer> shape, int length) {
        int expect = 1;
        for (Integer i : shape) {
            expect *= i;
        }
        if (expect == length) return;
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
        HeaderValue headerValue = new HeaderValue("I64", shape, dataOffsets);
        header.put(tensorName, headerValue);
        bodies.put(tensorName, longs);
    }

    public void add(String tensorName, List<Integer> shape, LongBuffer longBuffer) {
        checkLength(shape, longBuffer.limit());
        Map.Entry<Integer, Integer> dataOffsets;
        {
            int begin = byteSize;
            byteSize += Long.BYTES * (longBuffer.limit() - longBuffer.position());
            int end = byteSize;
            dataOffsets = new AbstractMap.SimpleEntry<>(begin, end);
        }
        HeaderValue headerValue = new HeaderValue("I64", shape, dataOffsets);
        header.put(tensorName, headerValue);
        bodies.put(tensorName, longBuffer);
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
        HeaderValue headerValue = new HeaderValue("F32", shape, dataOffsets);
        header.put(tensorName, headerValue);
        bodies.put(tensorName, floats);
    }

    public void add(String tensorName, List<Integer> shape, FloatBuffer floatBuffer) {
        checkLength(shape, floatBuffer.limit());
        Map.Entry<Integer, Integer> dataOffsets;
        {
            int begin = byteSize;
            byteSize += Float.BYTES * (floatBuffer.limit() - floatBuffer.position());
            int end = byteSize;
            dataOffsets = new AbstractMap.SimpleEntry<>(begin, end);
        }
        HeaderValue headerValue = new HeaderValue("F32", shape, dataOffsets);
        header.put(tensorName, headerValue);
        bodies.put(tensorName, floatBuffer);
    }

    public int contentLength() {
        return Long.BYTES + serializeHeader().getBytes(StandardCharsets.UTF_8).length + byteSize;
    }

    private String serializeHeader() {
        StringBuilder headerBuilder = new StringBuilder();
        headerBuilder.append('{');
        if (!header.isEmpty()) {
            for (Map.Entry<String, HeaderValue> entry : header.entrySet()) {
                headerBuilder.append('"');
                headerBuilder.append(entry.getKey());
                headerBuilder.append('"');
                headerBuilder.append(':');
                headerBuilder.append(entry.getValue().serialize());
                headerBuilder.append(',');
            }
            headerBuilder.deleteCharAt(headerBuilder.length() - 1);
        }
        headerBuilder.append('}');

        String stringHeader = headerBuilder.toString();
        int padding = 8 - (stringHeader.getBytes(StandardCharsets.UTF_8).length % 8);
        headerBuilder.append(" ".repeat(padding));
        return headerBuilder.toString();
    }

    private ByteBuffer serializeByteBuffer() {
        ByteBuffer byteBuffer = ByteBuffer.wrap(new byte[byteSize]);
        for (Map.Entry<String, HeaderValue> entry : header.entrySet()) {
            ByteBuffer bb;
            {
                Map.Entry<Integer, Integer> dataOffsets = entry.getValue().dataOffsets;
                int begin = dataOffsets.getKey();
                int end = dataOffsets.getValue();
                bb = ByteBuffer.wrap(byteBuffer.array(), begin, end - begin).order(ByteOrder.LITTLE_ENDIAN);
            }

            Object object = bodies.get(entry.getKey());
            if (object instanceof long[]) {
                bb.asLongBuffer().put((long[]) object);
                continue;
            }
            if (object instanceof LongBuffer) {
                bb.asLongBuffer().put((LongBuffer) object);
                continue;
            }
            if (object instanceof float[]) {
                bb.asFloatBuffer().put((float[]) object);
                continue;
            }
            if (object instanceof FloatBuffer) {
                bb.asFloatBuffer().put((FloatBuffer) object);
                continue;
            }
            throw new IllegalArgumentException("Unsupported type: " + object.getClass().getTypeName());
        }
        return byteBuffer;
    }

    public void save(File file) throws IOException {
        DataOutputStream dataOutputStream;
        {
            FileOutputStream fileOutputStream;
            try {
                fileOutputStream = new FileOutputStream(file);
            } catch (FileNotFoundException e) {
                throw new IOException(e);
            }
            BufferedOutputStream bufferedOutputStream = new BufferedOutputStream(fileOutputStream);
            dataOutputStream = new DataOutputStream(bufferedOutputStream);
        }
        save(dataOutputStream);
        dataOutputStream.close();
    }

    public void save(DataOutputStream dataOutputStream) throws IOException {
        String stringHeader = serializeHeader();
        {
            byte[] littleEndianBytesHeaderSize = new byte[Long.BYTES];
            long headerSize = stringHeader.getBytes(StandardCharsets.UTF_8).length;
            ByteBuffer.wrap(littleEndianBytesHeaderSize).order(ByteOrder.LITTLE_ENDIAN).asLongBuffer().put(headerSize);
            dataOutputStream.write(littleEndianBytesHeaderSize);
        }
        dataOutputStream.writeBytes(stringHeader);
        dataOutputStream.write(serializeByteBuffer().array());
    }
}
