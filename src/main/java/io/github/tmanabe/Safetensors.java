package io.github.tmanabe;

import javax.script.ScriptEngine;
import javax.script.ScriptEngineManager;
import javax.script.ScriptException;
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.nio.charset.StandardCharsets;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Safetensors {
    public static class HeaderValue {
        private final String dtype;
        private final List<Integer> shape;
        private final Map.Entry<Integer, Integer> dataOffsets;

        private static HeaderValue load(Map<?, ?> map) {
            assert map.containsKey("dtype");
            String dtype = map.get("dtype").toString();

            assert map.containsKey("shape");
            List<Integer> shape = new ArrayList<>();
            {
                Map<?, ?> m = (Map<?, ?>) map.get("shape");
                for (int i = 0; i < m.size(); ++i) {
                    assert m.containsKey(Integer.toString(i));
                    shape.add((Integer) m.get(Integer.toString(i)));
                }
            }

            assert map.containsKey("data_offsets");
            AbstractMap.SimpleEntry<Integer, Integer> dataOffsets;
            {
                Map<?, ?> m = (Map<?, ?>) map.get("data_offsets");
                assert 2 == m.size() && m.containsKey("0") && m.containsKey("1");
                Integer begin = (Integer) m.get("0"), end = (Integer) m.get("1");
                dataOffsets = new AbstractMap.SimpleEntry<>(begin, end);
            }

            assert 3 == map.size();
            return new HeaderValue(dtype, shape, dataOffsets);
        }

        HeaderValue(String dtype, List<Integer> shape, Map.Entry<Integer, Integer> dataOffsets) {
            this.dtype = dtype;
            this.shape = shape;
            this.dataOffsets = dataOffsets;
        }

        private String save() {
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

        public String getDtype() {
            return dtype;
        }

        public List<Integer> getShape() {
            return shape;
        }

        public Map.Entry<Integer, Integer> getDataOffsets() {
            return dataOffsets;
        }
    }

    private final Map<String, HeaderValue> header;
    private final ByteBuffer byteBuffer;

    public static Safetensors load(File file) throws IOException {
        DataInputStream dataInputStream;
        {
            FileInputStream fileInputStream = new FileInputStream(file);
            BufferedInputStream bufferedInputStream = new BufferedInputStream(fileInputStream);
            dataInputStream = new DataInputStream(bufferedInputStream);
        }

        long headerSize;
        {
            byte[] littleEndianBytesHeaderSize = new byte[8];
            int read = dataInputStream.read(littleEndianBytesHeaderSize);
            assert 8 == read;
            headerSize = ByteBuffer.wrap(littleEndianBytesHeaderSize).order(ByteOrder.LITTLE_ENDIAN).getLong();
        }

        String stringHeader;
        {
            assert headerSize <= Integer.MAX_VALUE;
            byte[] bytesHeader = new byte[(int) headerSize];
            int read = dataInputStream.read(bytesHeader);
            assert headerSize == read;
            stringHeader = new String(bytesHeader, StandardCharsets.UTF_8);
        }

        Object objectHeader;
        try {
            ScriptEngineManager scriptEngineManager = new ScriptEngineManager();
            ScriptEngine scriptEngine = scriptEngineManager.getEngineByName("JavaScript");
            objectHeader = scriptEngine.eval("var varHeader = " + stringHeader + "; varHeader;");
        } catch (ScriptException e) {
            throw new IOException(e);
        }

        Map<String, HeaderValue> header = new HashMap<>();
        for (Map.Entry<?, ?> entry : ((Map<?, ?>) objectHeader).entrySet()) {
            String tensorName = entry.getKey().toString();
            if (tensorName.equals("__metadata__")) continue;
            Map<?, ?> map = (Map<?, ?>) entry.getValue();
            header.put(tensorName, HeaderValue.load(map));
        }

        int byteBufferSize = 0;
        for (HeaderValue headerValue : header.values()) {
            byteBufferSize = Math.max(byteBufferSize, headerValue.getDataOffsets().getValue());
        }

        ByteBuffer byteBuffer;
        {
            byte[] bytes = new byte[byteBufferSize];
            int read = dataInputStream.read(bytes);
            assert byteBufferSize == read;
            byteBuffer = ByteBuffer.wrap(bytes);
        }

        dataInputStream.close();
        return new Safetensors(header, byteBuffer);
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

        StringBuilder headerBuilder = new StringBuilder();
        headerBuilder.append('{');
        if (!header.isEmpty()) {
            for (Map.Entry<String, HeaderValue> entry : header.entrySet()) {
                headerBuilder.append('"');
                headerBuilder.append(entry.getKey());
                headerBuilder.append('"');
                headerBuilder.append(':');
                headerBuilder.append(entry.getValue().save());
                headerBuilder.append(',');
            }
            headerBuilder.deleteCharAt(headerBuilder.length() - 1);
        }
        headerBuilder.append('}');

        String stringHeader = headerBuilder.toString();
        int padding = 8 - (stringHeader.getBytes(StandardCharsets.UTF_8).length % 8);
        for (int i = 0; i < padding; ++i) {
            headerBuilder.append(' ');
        }
        stringHeader = headerBuilder.toString();

        {
            byte[] littleEndianBytesHeaderSize = new byte[8];
            long headerSize = stringHeader.getBytes(StandardCharsets.UTF_8).length;
            ByteBuffer.wrap(littleEndianBytesHeaderSize).order(ByteOrder.LITTLE_ENDIAN).asLongBuffer().put(headerSize);
            dataOutputStream.write(littleEndianBytesHeaderSize);
        }

        dataOutputStream.writeBytes(stringHeader);
        dataOutputStream.write(byteBuffer.array());
        dataOutputStream.close();
    }

    Safetensors(Map<String, HeaderValue> header, ByteBuffer byteBuffer) {
        this.header = header;
        this.byteBuffer = byteBuffer;
    }

    private void checkContains(String tensorName) {
        if (header.containsKey(tensorName)) return;
        throw new IllegalArgumentException("Tensor not found: " + tensorName);
    }

    public Map<String, HeaderValue> getHeader() {
        return Collections.unmodifiableMap(header);
    }

    public ByteBuffer getByteBuffer(String tensorName) {
        checkContains(tensorName);
        HeaderValue headerValue = header.get(tensorName);
        Integer begin = headerValue.getDataOffsets().getKey(), end = headerValue.getDataOffsets().getValue();
        return ByteBuffer.wrap(byteBuffer.array(), begin, end - begin).order(ByteOrder.LITTLE_ENDIAN);
    }

    public LongBuffer getLongBuffer(String tensorName) {
        checkContains(tensorName);
        if (header.get(tensorName).getDtype().equals("I64")) {
            return getByteBuffer(tensorName).asLongBuffer();
        }
        throw new IllegalArgumentException("Unsupported dtype: " + header.get(tensorName).getDtype());
    }

    public FloatBuffer getFloatBuffer(String tensorName) {
        checkContains(tensorName);
        if (header.get(tensorName).getDtype().equals("F32")) {
            return getByteBuffer(tensorName).asFloatBuffer();
        }
        throw new IllegalArgumentException("Unsupported dtype: " + header.get(tensorName).getDtype());
    }
}
