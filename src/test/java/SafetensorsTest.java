import io.github.tmanabe.Safetensors;
import org.junit.Test;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.util.List;

public class SafetensorsTest {
    @Test
    public void testLoad() throws IOException {
        URL sampleURL = this.getClass().getResource("sample.safetensors");
        assert null != sampleURL;
        String sampleString = sampleURL.getFile();
        Safetensors sample = Safetensors.load(new File(sampleString));
        {
            String tensorName = "some_ints";
            {
                List<Integer> shape = sample.getHeader().get(tensorName).getShape();
                assert 2 == shape.size();
                assert 1 == shape.get(0);
                assert 4 == shape.get(1);
            }
            {
                LongBuffer longBuffer = sample.getLongBuffer(tensorName);
                assert 4 == longBuffer.limit();
                assert -1L == longBuffer.get(0);
                assert 0L == longBuffer.get(1);
                assert 1L == longBuffer.get(2);
                assert 2L == longBuffer.get(3);
            }
        }
        {
            String tensorName = "some_floats";
            {
                List<Integer> shape = sample.getHeader().get(tensorName).getShape();
                assert 3 == shape.size();
                assert 1 == shape.get(0);
                assert 2 == shape.get(1);
                assert 2 == shape.get(2);
            }
            {
                FloatBuffer floatBuffer = sample.getFloatBuffer(tensorName);
                assert 4 == floatBuffer.limit();
                assert -1.0f == floatBuffer.get(0);
                assert 0.0f == floatBuffer.get(1);
                assert 1.0f == floatBuffer.get(2);
                assert 2.0f == floatBuffer.get(3);
            }
        }
    }
}
