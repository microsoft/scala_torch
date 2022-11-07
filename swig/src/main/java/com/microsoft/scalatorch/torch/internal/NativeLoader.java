package com.microsoft.scalatorch.torch.internal;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.channels.FileChannel;
import java.nio.channels.Channels;
import java.nio.channels.ReadableByteChannel;
import java.nio.file.Path;

/** Much like https://github.com/sbt/sbt-jni or https://github.com/scijava/native-lib-loader,
 * but both of them are a little too opinionated. */
public class NativeLoader {

    /**
     * Extracts the library given by the path elem +: elems. For example,
     * {{{extractAndLoadLibrary(Paths.get("tmp", true, "native", "c10"}}} will look for a resource
     * called `native/libc10.dylib` on a mac and `native/c10.dll` on windows.
     *
     * @param dir       The directory to extract the native library to.
     * @param isLibName If true, the last element of the path is treated as a library name and [[System.mapLibraryName]].
     *                  Otherwise, it is taken as is, so it must have the correct extensions (e.g. libfoo.so.1).
     *                  will be called on it first.
     * @return The extracted file, or null if no resource could be found.
     * @throws IOException Also might throw other RuntimeExceptions if library loading fails.
     */
    public static File extractAndLoadLibrary(Path dir, boolean isLibName, String elem, String... elems) throws IOException {
        String maybeName = (elems.length == 0) ? elem : elems[elems.length - 1];
        String libName = isLibName ? System.mapLibraryName(maybeName) : maybeName;
        String[] resourcePathElems = new String[elems.length + 1];
        if (elems.length > 0) {
            resourcePathElems[0] = elem;
            System.arraycopy(elems, 0, resourcePathElems, 1, elems.length - 1);
        }
        resourcePathElems[elems.length] = libName;
        String resourcePath = String.join("/", resourcePathElems);
        File result = extract(dir, resourcePath);
        if (result == null) return null;
        System.load(result.toString());
        return result;
    }

    private static File extract(Path dir, String resourcePath) throws IOException {
        URL url = NativeLoader.class.getResource("/" + resourcePath);
        if (url == null) return null;

        try(InputStream in = NativeLoader.class.getResourceAsStream("/" + resourcePath)) {
            File file = file(dir, resourcePath);
            file.deleteOnExit();

            ReadableByteChannel src = Channels.newChannel(in);
            try (FileChannel dest = new FileOutputStream(file).getChannel()) {
                dest.transferFrom(src, 0, Long.MAX_VALUE);

                return file;
            }
        }
    }

    private static File file(Path dir, String path) throws IOException {
        String name = new File(path).getName();

        File file = dir.resolve(name).toFile();
        if (file.exists() && !file.isFile())
            throw new IllegalArgumentException(file.getAbsolutePath() + " is not a file.");
        if (!file.exists()) file.createNewFile();
        return file;
    }
}
