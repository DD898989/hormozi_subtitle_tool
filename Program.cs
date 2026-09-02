using Microsoft.AspNetCore.Http.Features;
using System.Diagnostics;
using System.Text;
using System.IO.Compression;
using Whisper.net;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen(options =>
{
    options.SwaggerDoc("v1", new() { Title = "Hormozi Subtitle Tool API", Version = "v1" });
});

builder.Services.Configure<FormOptions>(options =>
{
    options.MultipartBodyLengthLimit = 500 * 1024 * 1024;
});
builder.WebHost.ConfigureKestrel(options =>
{
    options.Limits.MaxRequestBodySize = 500 * 1024 * 1024;
});

var app = builder.Build();

app.UseSwagger();
app.UseSwaggerUI(options =>
{
    options.SwaggerEndpoint("/swagger/v1/swagger.json", "v1");
    options.RoutePrefix = string.Empty;
});

app.MapPost("/api/caption", async (IFormFile file, IFormFile srtFile) =>
{
    var tempMp4 = Path.Combine(Path.GetTempPath(), Guid.NewGuid() + ".mp4");
    var tempWav = Path.Combine(Path.GetTempPath(), Guid.NewGuid() + ".wav");
    var modelPath = "ggml-base.bin";


    using (var stream = File.Create(tempMp4))
    {
        await file.CopyToAsync(stream);
    }

    RunProcess(Env.FFMPEG_PATH, $"-y -i \"{tempMp4}\" -ar 16000 -ac 1 -c:a pcm_s16le \"{tempWav}\"");

    if (!File.Exists(modelPath))
    {
        using var client = new HttpClient();
        using var request = new HttpRequestMessage(HttpMethod.Get, "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin");
        using var response = await client.SendAsync(request, HttpCompletionOption.ResponseHeadersRead);
        response.EnsureSuccessStatusCode();
        using var fileStream = File.Create(modelPath);
        using var contentStream = await response.Content.ReadAsStreamAsync();
        await contentStream.CopyToAsync(fileStream);
    }

    using var whisperFactory = WhisperFactory.FromPath(modelPath);
    using var processor = whisperFactory.CreateBuilder()
        .WithLanguage("en")
        .WithTokenTimestamps()
        .Build();

    using var wavStream = File.OpenRead(tempWav);

    var assHeader = "[Script Info]\nScriptType: v4.00+\nPlayResX: 1080\nPlayResY: 1920\nScaledBorderAndShadow: yes\n\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\nStyle: Default,Arial,80,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,6,3,2,40,40,250,1\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n";
    var assOriginal = new StringBuilder(assHeader);
    var assShort = new StringBuilder(assHeader);
    var assCorrect = new StringBuilder(assHeader);
    var assCorrectShortBounce = new StringBuilder(assHeader);

    var segments = new List<SegmentData>();
    await foreach (var segment in processor.ProcessAsync(wavStream))
        segments.Add(segment);

    // Read and parse SRT file
    string srtContent;
    using (var srtStream = srtFile.OpenReadStream())
    using (var reader = new StreamReader(srtStream, Encoding.UTF8))
    {
        srtContent = await reader.ReadToEndAsync();
    }
    var srtItems = ParseSrt(srtContent);

    // Flatten Whisper tokens
    var allWhisperTokens = new List<WhisperToken>();
    for (int segIdx = 0; segIdx < segments.Count; segIdx++)
    {
        var segment = segments[segIdx];
        var tokens = segment.Tokens
            .Select(t => (
                Text: t.Text?.Trim(' ', '.', ',', '!', '?', '"', '\'', '(', ')', '[', ']') ?? string.Empty,
                Start: TimeSpan.FromMilliseconds(t.Start * 10),
                End: TimeSpan.FromMilliseconds(t.End * 10)
            ))
            .Where(t => !string.IsNullOrEmpty(t.Text))
            .ToArray();

        if (tokens.Length > 0 && tokens.Last().Text.Contains("_TT_"))
            tokens = tokens.SkipLast(1).ToArray();

        // 1. 原本邏輯的 ASS
        for (int i = 0; i < tokens.Length; i++)
        {
            var cur = tokens[i];
            var start = FormatTime(cur.Start);
            var end = FormatTime(i < tokens.Length - 1 ? tokens[i + 1].Start : segment.End);
            var text = string.Join(" ", tokens.Select((t, idx) => idx == i ? $"{{\\c&H00FFFF&}}{t.Text}{{\\r}}" : t.Text));
            assOriginal.AppendLine($"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}");
        }

        // 2. 最多 3 個字一句的 ASS
        for (int g = 0; g < tokens.Length; g += 3)
        {
            var group = tokens.Skip(g).Take(3).ToArray();
            for (int i = 0; i < group.Length; i++)
            {
                var cur = group[i];
                var start = FormatTime(cur.Start);
                var end = FormatTime(g + i < tokens.Length - 1 ? tokens[g + i + 1].Start : segment.End);
                var text = string.Join(" ", group.Select((t, idx) => idx == i ? $"{{\\c&H00FFFF&}}{t.Text}{{\\r}}" : t.Text));
                assShort.AppendLine($"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}");
            }
        }

        // Store to allWhisperTokens for correction alignment
        allWhisperTokens.AddRange(tokens.Select(t => new WhisperToken { Text = t.Text, Start = t.Start, End = t.End, SegmentIndex = segIdx }));
    }

    // 3. 跟正確的srt做校正的ASS
    {
        var allCorrectedTokens = new List<WhisperToken>();
        var tokenGroups = srtItems.ToDictionary(item => item, _ => new List<WhisperToken>());
        foreach (var token in allWhisperTokens)
        {
            var tokenMid = token.Start + (token.End - token.Start) / 2;
            SrtItem? bestItem = null;
            double minDistanceMs = double.MaxValue;

            foreach (var item in srtItems)
            {
                if (tokenMid >= item.Start && tokenMid <= item.End)
                {
                    bestItem = item;
                    break;
                }
                double dist;
                if (tokenMid < item.Start)
                    dist = (item.Start - tokenMid).TotalMilliseconds;
                else
                    dist = (tokenMid - item.End).TotalMilliseconds;

                if (dist < minDistanceMs)
                {
                    minDistanceMs = dist;
                    bestItem = item;
                }
            }

            if (bestItem != null)
            {
                tokenGroups[bestItem].Add(token);
            }
        }

        foreach (var item in srtItems)
        {
            var A = item.Words;
            var B = tokenGroups[item];
            int n = A.Count;
            int m = B.Count;

            if (n == 0) continue;

            var srtItemCorrected = new List<WhisperToken>();

            if (m == 0)
            {
                var duration = item.End - item.Start;
                var step = duration / (n + 1);
                
                // Find the closest Whisper segment index
                int bestSegIdx = 0;
                double minSegDistMs = double.MaxValue;
                for (int s = 0; s < segments.Count; s++)
                {
                    var seg = segments[s];
                    if (item.Start >= seg.Start && item.Start <= seg.End)
                    {
                        bestSegIdx = s;
                        break;
                    }
                    double dist = Math.Min(
                        Math.Abs((seg.Start - item.Start).TotalMilliseconds),
                        Math.Abs((seg.End - item.Start).TotalMilliseconds)
                    );
                    if (dist < minSegDistMs)
                    {
                        minSegDistMs = dist;
                        bestSegIdx = s;
                    }
                }

                for (int u = 0; u < n; u++)
                {
                    srtItemCorrected.Add(new WhisperToken
                    {
                        Text = A[u],
                        Start = item.Start + step * (u + 1),
                        End = item.Start + step * (u + 2),
                        SegmentIndex = bestSegIdx
                    });
                }
            }
            else
            {
                double[,] dp = new double[n + 1, m + 1];
                int[,] parent = new int[n + 1, m + 1];

                for (int i = 0; i <= n; i++)
                {
                    for (int j = 0; j <= m; j++)
                    {
                        if (i == 0 && j == 0)
                        {
                            dp[i, j] = 0;
                        }
                        else if (i == 0)
                        {
                            dp[i, j] = dp[i, j - 1] + 1.5;
                            parent[i, j] = 2;
                        }
                        else if (j == 0)
                        {
                            dp[i, j] = dp[i - 1, j] + 1.5;
                            parent[i, j] = 3;
                        }
                        else
                        {
                            double costMatch = dp[i - 1, j - 1] + 2.0 * (1.0 - WordSimilarity(A[i - 1], B[j - 1].Text));
                            double costSkipToken = dp[i, j - 1] + 1.5;
                            double costSkipWord = dp[i - 1, j] + 1.5;

                            double minCost = costMatch;
                            int p = 1;

                            if (costSkipToken < minCost)
                            {
                                minCost = costSkipToken;
                                p = 2;
                            }
                            if (costSkipWord < minCost)
                            {
                                minCost = costSkipWord;
                                p = 3;
                            }

                            dp[i, j] = minCost;
                            parent[i, j] = p;
                        }
                    }
                }

                int currI = n;
                int currJ = m;
                var alignment = new List<(int wordIdx, int tokenIdx)>();

                while (currI > 0 || currJ > 0)
                {
                    int p = parent[currI, currJ];
                    if (p == 1)
                    {
                        alignment.Add((currI - 1, currJ - 1));
                        currI--;
                        currJ--;
                    }
                    else if (p == 2)
                    {
                        alignment.Add((-1, currJ - 1));
                        currJ--;
                    }
                    else
                    {
                        alignment.Add((currI - 1, -1));
                        currI--;
                    }
                }
                alignment.Reverse();

                int k = 0;
                while (k < alignment.Count)
                {
                    var pair = alignment[k];
                    if (pair.wordIdx >= 0 && pair.tokenIdx >= 0)
                    {
                        srtItemCorrected.Add(new WhisperToken
                        {
                            Text = A[pair.wordIdx],
                            Start = B[pair.tokenIdx].Start,
                            End = B[pair.tokenIdx].End,
                            SegmentIndex = B[pair.tokenIdx].SegmentIndex
                        });
                        k++;
                    }
                    else if (pair.wordIdx == -1 && pair.tokenIdx >= 0)
                    {
                        k++;
                    }
                    else if (pair.wordIdx >= 0 && pair.tokenIdx == -1)
                    {
                        var unmatchedWords = new List<string>();
                        int startK = k;
                        while (k < alignment.Count && alignment[k].wordIdx >= 0 && alignment[k].tokenIdx == -1)
                        {
                            unmatchedWords.Add(A[alignment[k].wordIdx]);
                            k++;
                        }

                        TimeSpan prevEnd = item.Start;
                        int segmentIndex = -1;
                        if (srtItemCorrected.Count > 0)
                        {
                            prevEnd = srtItemCorrected.Last().End;
                            segmentIndex = srtItemCorrected.Last().SegmentIndex;
                        }
                        else
                        {
                            for (int prevK = startK - 1; prevK >= 0; prevK--)
                            {
                                if (alignment[prevK].tokenIdx >= 0)
                                {
                                    prevEnd = B[alignment[prevK].tokenIdx].End;
                                    segmentIndex = B[alignment[prevK].tokenIdx].SegmentIndex;
                                    break;
                                }
                            }
                        }

                        TimeSpan nextStart = item.End;
                        int nextSegmentIndex = -1;
                        for (int nextK = k; nextK < alignment.Count; nextK++)
                        {
                            if (alignment[nextK].tokenIdx >= 0)
                            {
                                nextStart = B[alignment[nextK].tokenIdx].Start;
                                nextSegmentIndex = B[alignment[nextK].tokenIdx].SegmentIndex;
                                break;
                            }
                        }

                        if (segmentIndex == -1)
                        {
                            segmentIndex = nextSegmentIndex != -1 ? nextSegmentIndex : 0;
                        }

                        if (nextStart < prevEnd) nextStart = prevEnd;

                        var duration = nextStart - prevEnd;
                        var step = duration / (unmatchedWords.Count + 1);

                        for (int u = 0; u < unmatchedWords.Count; u++)
                        {
                            srtItemCorrected.Add(new WhisperToken
                            {
                                Text = unmatchedWords[u],
                                Start = prevEnd + step * (u + 1),
                                End = prevEnd + step * (u + 2),
                                SegmentIndex = segmentIndex
                            });
                        }
                    }
                }
            }

            allCorrectedTokens.AddRange(srtItemCorrected);
        }

        // Generate assCorrect lines by grouping by SegmentIndex to match original.ass line boundaries exactly
        var correctedGroups = allCorrectedTokens
            .GroupBy(t => t.SegmentIndex)
            .ToDictionary(g => g.Key, g => g.OrderBy(t => t.Start).ToList());

        for (int segIdx = 0; segIdx < segments.Count; segIdx++)
        {
            var segment = segments[segIdx];
            if (!correctedGroups.TryGetValue(segIdx, out var groupTokens) || groupTokens.Count == 0)
                continue;

            for (int i = 0; i < groupTokens.Count; i++)
            {
                var cur = groupTokens[i];
                var start = FormatTime(cur.Start);
                var end = FormatTime(i < groupTokens.Count - 1 ? groupTokens[i + 1].Start : segment.End);
                var text = string.Join(" ", groupTokens.Select((t, idx) => idx == i ? $"{{\\c&H00FFFF&}}{t.Text}{{\\r}}" : t.Text));
                assCorrect.AppendLine($"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}");
            }
        }

        // Generate assCorrectShortBounce lines (corrected, max 3 words, large center, random bounce, no highlighting)
        var rand = new Random();
        for (int g = 0; g < allCorrectedTokens.Count; g += 3)
        {
            var group = allCorrectedTokens.Skip(g).Take(3).ToArray();
            if (group.Length == 0) continue;

            var start = FormatTime(group[0].Start);
            TimeSpan groupEnd;
            if (g + 3 < allCorrectedTokens.Count)
            {
                groupEnd = allCorrectedTokens[g + 3].Start;
            }
            else
            {
                groupEnd = group.Last().End;
            }
            var end = FormatTime(groupEnd);

            var text = string.Join(" ", group.Select(t => t.Text));
            int posX = 540 + rand.Next(-40, 41);
            int posY = 960 + rand.Next(-60, 61);
            int rot = rand.Next(-5, 6);

            assCorrectShortBounce.AppendLine($"Dialogue: 0,{start},{end},Default,,0,0,0,,{{\\an5\\fs130\\pos({posX},{posY})\\frz{rot}}}{text}");
        }
    }


    var zipMs = new MemoryStream();
    using (var archive = new ZipArchive(zipMs, ZipArchiveMode.Create, leaveOpen: true))
    {
        // 寫入原本 ASS
        var entry1 = archive.CreateEntry("original.ass");
        using (var writer = new StreamWriter(entry1.Open(), Encoding.UTF8))
            await writer.WriteAsync(assOriginal.ToString());

        // 寫入 3 字 ASS
        var entry2 = archive.CreateEntry("max_3_words.ass");
        using (var writer = new StreamWriter(entry2.Open(), Encoding.UTF8))
            await writer.WriteAsync(assShort.ToString());

        // 寫入校正後的 ASS (3字黃色高亮樣式)
        var entryCorrect = archive.CreateEntry("assCorrect.ass");
        using (var writer = new StreamWriter(entryCorrect.Open(), Encoding.UTF8))
            await writer.WriteAsync(assCorrect.ToString());

        // 寫入校正後 3 字隨機跳動大字幕 ASS
        var entryCorrectBounce = archive.CreateEntry("corrected_max_3_words_bounce.ass");
        using (var writer = new StreamWriter(entryCorrectBounce.Open(), Encoding.UTF8))
            await writer.WriteAsync(assCorrectShortBounce.ToString());

        // 寫入原本影片
        var entry3 = archive.CreateEntry(file.FileName ?? "video.mp4");
        using (var videoStream = File.OpenRead(tempMp4))
        using (var entryStream = entry3.Open())
            await videoStream.CopyToAsync(entryStream);
    }

    zipMs.Position = 0;
    return Results.File(zipMs, "application/zip", "captions.zip");
})
.DisableAntiforgery();

app.Run();

static void RunProcess(string cmd, string args)
{
    using var proc = Process.Start(new ProcessStartInfo
    {
        FileName = cmd,
        Arguments = args,
        RedirectStandardError = true,
        UseShellExecute = false,
        CreateNoWindow = true
    });
    proc!.WaitForExit();
}

static string FormatTime(TimeSpan t) => $"{t.Hours}:{t.Minutes:D2}:{t.Seconds:D2}.{t.Milliseconds / 10:D2}";





static List<SrtItem> ParseSrt(string srtContent)
{
    var blocks = srtContent.Split(new[] { "\r\n\r\n", "\n\n", "\r\r" }, StringSplitOptions.RemoveEmptyEntries);
    var srtItems = new List<SrtItem>();
    foreach (var block in blocks)
    {
        var lines = block.Split(new[] { "\r\n", "\r", "\n" }, StringSplitOptions.RemoveEmptyEntries)
                         .Select(l => l.Trim())
                         .Where(l => !string.IsNullOrEmpty(l))
                         .ToArray();
        if (lines.Length >= 2)
        {
            string timeLine = "";
            int textStartIndex = 0;
            if (lines[0].Contains("-->"))
            {
                timeLine = lines[0];
                textStartIndex = 1;
            }
            else if (lines.Length >= 2 && lines[1].Contains("-->"))
            {
                timeLine = lines[1];
                textStartIndex = 2;
            }

            if (!string.IsNullOrEmpty(timeLine))
            {
                ParseTimeLine(timeLine, out var start, out var end);
                var textLines = lines.Skip(textStartIndex);
                var textStr = string.Join(" ", textLines);
                srtItems.Add(new SrtItem
                {
                    Start = start,
                    End = end,
                    Text = textStr,
                    Words = SplitIntoWords(textStr)
                });
            }
        }
    }
    return srtItems;
}

static void ParseTimeLine(string line, out TimeSpan start, out TimeSpan end)
{
    start = TimeSpan.Zero;
    end = TimeSpan.Zero;
    var parts = line.Split("-->", StringSplitOptions.TrimEntries);
    if (parts.Length == 2)
    {
        TimeSpan.TryParse(parts[0].Replace(',', '.'), out start);
        TimeSpan.TryParse(parts[1].Replace(',', '.'), out end);
    }
}

static List<string> SplitIntoWords(string text)
{
    return text.Split(new[] { ' ', '\t', '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries)
               .Select(w => w.Trim(' ', '.', ',', '!', '?', '"', '\'', '(', ')', '[', ']', '-'))
               .Where(w => !string.IsNullOrEmpty(w))
               .ToList();
}

static int LevenshteinDistance(string s, string t)
{
    if (string.IsNullOrEmpty(s)) return t?.Length ?? 0;
    if (string.IsNullOrEmpty(t)) return s.Length;
    int n = s.Length;
    int m = t.Length;
    int[,] d = new int[n + 1, m + 1];
    for (int i = 0; i <= n; d[i, 0] = i++) ;
    for (int j = 0; j <= m; d[0, j] = j++) ;
    for (int i = 1; i <= n; i++)
    {
        for (int j = 1; j <= m; j++)
        {
            int cost = (t[j - 1] == s[i - 1]) ? 0 : 1;
            d[i, j] = Math.Min(Math.Min(d[i - 1, j] + 1, d[i, j - 1] + 1), d[i - 1, j - 1] + cost);
        }
    }
    return d[n, m];
}

static double WordSimilarity(string w1, string w2)
{
    if (w1 == w2) return 1.0;
    int maxLen = Math.Max(w1.Length, w2.Length);
    if (maxLen == 0) return 1.0;
    int dist = LevenshteinDistance(w1, w2);
    return 1.0 - (double)dist / maxLen;
}

class SrtItem
{
    public TimeSpan Start { get; set; }
    public TimeSpan End { get; set; }
    public string Text { get; set; } = string.Empty;
    public List<string> Words { get; set; } = new();
}

class WhisperToken
{
    public string Text { get; set; } = string.Empty;
    public TimeSpan Start { get; set; }
    public TimeSpan End { get; set; }
    public int SegmentIndex { get; set; } = -1;
}
