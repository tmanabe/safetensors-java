package io.github.tmanabe;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
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

public class SafetensorsViewer {
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

        private HeaderValue(String dtype, List<Integer> shape, Map.Entry<Integer, Integer> dataOffsets) {
            this.dtype = dtype;
            this.shape = shape;
            this.dataOffsets = dataOffsets;
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

    public static SafetensorsViewer load(File file) throws IOException {
        DataInputStream dataInputStream;
        {
            FileInputStream fileInputStream = new FileInputStream(file);
            BufferedInputStream bufferedInputStream = new BufferedInputStream(fileInputStream);
            dataInputStream = new DataInputStream(bufferedInputStream);
        }
        SafetensorsViewer safetensorsViewer = load(dataInputStream);
        dataInputStream.close();
        return safetensorsViewer;
    }

    public static SafetensorsViewer load(DataInputStream dataInputStream) throws IOException {
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

        return new SafetensorsViewer(header, byteBuffer);
    }

    private SafetensorsViewer(Map<String, HeaderValue> header, ByteBuffer byteBuffer) {
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
