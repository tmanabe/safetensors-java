package io.github.tmanabe;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

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
import java.util.Iterator;
import java.util.List;
import java.util.Map;

public class Safetensors {
    public static class HeaderValue {
        private final String dtype;
        private final List<Integer> shape;
        private final Map.Entry<Integer, Integer> dataOffsets;

        private static HeaderValue load(JsonNode jsonNode) {
            assert jsonNode.has("dtype");
            String dtype = jsonNode.get("dtype").asText();

            assert jsonNode.has("shape");
            List<Integer> shape = new ArrayList<>();
            {
                JsonNode jsonNodeShape = jsonNode.get("shape");
                assert jsonNodeShape.isArray();
                for (int i = 0; i < jsonNodeShape.size(); ++i) {
                    shape.add(jsonNodeShape.get(i).asInt());
                }
            }

            assert jsonNode.has("data_offsets");
            AbstractMap.SimpleEntry<Integer, Integer> dataOffsets;
            {
                JsonNode jsonNodeDataOffsets = jsonNode.get("data_offsets");
                assert jsonNodeDataOffsets.isArray();
                assert 2 == jsonNodeDataOffsets.size();
                Integer begin = jsonNodeDataOffsets.get(0).asInt(), end = jsonNodeDataOffsets.get(1).asInt();
                dataOffsets = new AbstractMap.SimpleEntry<>(begin, end);
            }

            assert 3 == jsonNode.size();
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
        Safetensors safetensors = load(dataInputStream);
        dataInputStream.close();
        return safetensors;
    }

    public static Safetensors load(DataInputStream dataInputStream) throws IOException {
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

        JsonNode jsonNodeHeader = new ObjectMapper().readTree(stringHeader);
        Map<String, HeaderValue> header = new HashMap<>();
        Iterator<Map.Entry<String, JsonNode>> iterator = jsonNodeHeader.fields();
        while (iterator.hasNext()) {
            Map.Entry<String, JsonNode> entry = iterator.next();
            String tensorName = entry.getKey();
            if (tensorName.equals("__metadata__")) continue;
            JsonNode jsonNode = entry.getValue();
            header.put(tensorName, HeaderValue.load(jsonNode));
        }

        int byteBufferSize = 0;
        for (HeaderValue headerValue : header.values()) {
            byteBufferSize = Math.max(byteBufferSize, headerValue.getDataOffsets().getValue());
        }

        ByteBuffer byteBuffer;
        {
            byte[] bytes = new byte[byteBufferSize];
            int read = 0, total = 0;
            while(0 <= read){
                read = dataInputStream.read(bytes, total, bytes.length - total);
                total += read;
                if (bytes.length == total) break;
            }
            assert byteBufferSize == total;
            byteBuffer = ByteBuffer.wrap(bytes);
        }

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
        save(dataOutputStream);
        dataOutputStream.close();
    }

    static String save(Map<String, HeaderValue> header) {
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
        return headerBuilder.toString();
    }

    public void save(DataOutputStream dataOutputStream) throws IOException {
        String stringHeader = save(header);
        {
            byte[] littleEndianBytesHeaderSize = new byte[Long.BYTES];
            long headerSize = stringHeader.getBytes(StandardCharsets.UTF_8).length;
            ByteBuffer.wrap(littleEndianBytesHeaderSize).order(ByteOrder.LITTLE_ENDIAN).asLongBuffer().put(headerSize);
            dataOutputStream.write(littleEndianBytesHeaderSize);
        }

        dataOutputStream.writeBytes(stringHeader);
        dataOutputStream.write(byteBuffer.array());
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
