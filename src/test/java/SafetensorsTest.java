import io.github.tmanabe.SafetensorsViewer;
import io.github.tmanabe.SafetensorsBuilder;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.util.Arrays;
import java.util.List;

public class SafetensorsTest {
    public void assertBasic(SafetensorsViewer sample) {
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
    }

    @Test
    public void testLoad() throws IOException {
        URL url = this.getClass().getResource("sample.safetensors");
        assert null != url;
        File file = new File(url.getFile());
        SafetensorsViewer sample = SafetensorsViewer.load(file);
        assertBasic(sample);
    }

    @Rule
    public TemporaryFolder temporaryFolder = new TemporaryFolder();

    @Test
    public void testBuildSaveAndLoad() throws IOException {
        File file = temporaryFolder.newFile("subject.safetensors");
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
            safetensorsBuilder.save(file);
        }
        {
            SafetensorsViewer subject = SafetensorsViewer.load(file);
            assertBasic(subject);
        }
    }
}
