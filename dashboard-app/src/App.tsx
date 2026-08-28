import {
  Activity,
  BarChart3,
  Check,
  ChevronLeft,
  ChevronRight,
  CircleHelp,
  ExternalLink,
  Filter,
  Layers3,
  Menu,
  MousePointerClick,
  Play,
  Search,
  Sparkles,
  Table2,
  X,
} from "lucide-react";
import {
  Area,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  flexRender,
  getCoreRowModel,
  type ColumnDef,
  useReactTable,
} from "@tanstack/react-table";
import { useCallback, useEffect, useMemo, useState, type ReactNode } from "react";
import type { Snapshot, Topic, VideoDetail, VideoRow } from "./types";

const API_BASE = import.meta.env.VITE_API_BASE || "";
const palette = ["#3c2de7", "#9fd77b", "#f2ed58", "#e58ab4", "#7d84e8", "#56bcb0"];
const badgePalette: Record<string, string> = {
  "#60a5fa": "#3c2de7",
  "#34d399": "#376d22",
  "#fbbf24": "#675f00",
  "#fb7185": "#9c315d",
  "#c084fc": "#5d43b5",
  "#2dd4bf": "#14766e",
  "#67e8f9": "#3c2de7",
  "#a7f3d0": "#376d22",
  "#fde68a": "#675f00",
  "#c4b5fd": "#5d43b5",
};

const cn = (...parts: (string | false | null | undefined)[]) => parts.filter(Boolean).join(" ");
const apiUrl = (path: string) => API_BASE + path;

async function fetchJson<T>(path: string): Promise<T> {
  const response = await fetch(apiUrl(path));
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.error || "Request failed (" + response.status + ")");
  }
  return response.json() as Promise<T>;
}

function formatNumber(value: number | null | undefined, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return new Intl.NumberFormat("en", {
    notation: Math.abs(value) >= 1000 ? "compact" : "standard",
    maximumFractionDigits: digits,
  }).format(value);
}

function formatPercent(value: number | null | undefined) {
  if (value === null || value === undefined) return "—";
  return (value * 100).toFixed(value < 0.1 ? 1 : 0) + "%";
}

function formatDate(value: string | null | undefined) {
  if (!value) return "Unknown date";
  return new Intl.DateTimeFormat("en", { month: "short", year: "numeric" }).format(new Date(value));
}

function formatDuration(seconds: number | null) {
  if (seconds === null || seconds === undefined) return "—";
  return Math.floor(seconds / 60) + ":" + String(Math.round(seconds % 60)).padStart(2, "0");
}

function labelFor(value: string | null) {
  if (!value) return "Unclassified";
  if (value === "relevant") return "Accepted";
  if (value === "irrelevant") return "Out of scope";
  if (value === "needs_transcript") return "Needs transcript";
  if (value === "deferred_budget") return "Budget deferred";
  return value.replaceAll("_", " ");
}

function SectionHeading({ eyebrow, title, detail, icon }: { eyebrow: string; title: string; detail?: string; icon?: ReactNode }) {
  return <div className="section-heading mb-5 flex items-end justify-between gap-4"><div><div className="mono mb-2 flex items-center gap-2 text-[10px] font-medium uppercase tracking-[0.2em] text-cyan-300/75">{icon}{eyebrow}</div><h2 className="text-xl font-extrabold tracking-tight text-slate-50">{title}</h2></div>{detail && <p className="max-w-sm text-right text-xs leading-5 text-slate-400">{detail}</p>}</div>;
}

function Panel({ children, className = "" }: { children: ReactNode; className?: string }) {
  return <div className={cn("panel surface rounded-2xl p-5", className)}>{children}</div>;
}

function Badge({ children, color = "#60a5fa" }: { children: ReactNode; color?: string }) {
  const themeColor = badgePalette[color] || color;
  return <span className="badge inline-flex max-w-full items-center rounded-full border px-2 py-1 text-[10px] font-bold" style={{ color: themeColor, borderColor: themeColor + "45", background: themeColor + "12" }}>{children}</span>;
}

function MetricCard({ label, value, note, accent, icon }: { label: string; value: string; note: string; accent: string; icon: ReactNode }) {
  return <div className="metric-card surface relative overflow-hidden rounded-2xl p-5"><div className="absolute -right-8 -top-8 h-24 w-24 rounded-full opacity-20 blur-2xl" style={{ background: accent }} /><div className="relative flex items-start justify-between gap-2"><div><p className="text-xs font-semibold uppercase tracking-[0.12em] text-slate-400">{label}</p><p className="mt-3 text-3xl font-extrabold tracking-tight text-white">{value}</p><p className="mt-1 text-xs text-slate-400">{note}</p></div><span className="rounded-xl p-2.5" style={{ color: accent, background: accent + "18" }}>{icon}</span></div></div>;
}

function MiniStat({ label, value }: { label: string; value: string }) {
  return <div className="mini-stat rounded-xl border border-white/5 bg-white/[0.03] p-3"><p className="mono text-[10px] uppercase tracking-[0.12em] text-slate-600">{label}</p><p className="mt-2 text-sm font-bold text-slate-200">{value}</p></div>;
}

function ChartTooltip({ active, payload, label }: { active?: boolean; payload?: any[]; label?: string }) {
  if (!active || !payload?.length) return null;
  return <div className="chart-tooltip surface rounded-xl px-3 py-2 text-xs shadow-xl"><p className="mb-1 font-semibold text-slate-200">{label}</p>{payload.map((entry) => <p key={entry.dataKey} style={{ color: entry.color }} className="mono">{entry.name || entry.dataKey}: {formatNumber(entry.value)}</p>)}</div>;
}

export default function App() {
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [live, setLive] = useState(true);
  const [activeTopic, setActiveTopic] = useState("");
  const [search, setSearch] = useState("");
  const [label, setLabel] = useState("");
  const [channel, setChannel] = useState("");
  const [sort, setSort] = useState("views");
  const [page, setPage] = useState(0);
  const [videoData, setVideoData] = useState<{ rows: VideoRow[]; total: number } | null>(null);
  const [videoLoading, setVideoLoading] = useState(false);
  const [selectedVideo, setSelectedVideo] = useState<VideoDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [mobileMenu, setMobileMenu] = useState(false);

  const loadSnapshot = useCallback(async () => {
    try {
      setSnapshot(await fetchJson<Snapshot>("/api/snapshot"));
      setError(null);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not load dashboard");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadSnapshot();
    const interval = window.setInterval(async () => {
      try {
        const health = await fetchJson<{ fingerprint: string }>("/api/health");
        setLive(true);
        if (health.fingerprint !== snapshot?.server.fingerprint) void loadSnapshot();
      } catch {
        setLive(false);
      }
    }, 10_000);
    return () => window.clearInterval(interval);
  }, [loadSnapshot, snapshot?.server.fingerprint]);

  useEffect(() => {
    if (!snapshot) return;
    const params = new URLSearchParams({ limit: "50", offset: String(page * 50), sort, direction: "desc" });
    if (search) params.set("search", search);
    if (activeTopic) params.set("topic", activeTopic);
    if (label) params.set("label", label);
    if (channel) params.set("channel", channel);
    setVideoLoading(true);
    void fetchJson<{ rows: VideoRow[]; total: number }>("/api/videos?" + params.toString())
      .then(setVideoData)
      .catch((cause) => setError(cause instanceof Error ? cause.message : "Could not load videos"))
      .finally(() => setVideoLoading(false));
  }, [snapshot, page, search, activeTopic, label, channel, sort]);

  const openVideo = async (videoId: string) => {
    setDetailLoading(true);
    try {
      setSelectedVideo(await fetchJson<VideoDetail>("/api/videos/" + encodeURIComponent(videoId)));
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not load video");
    } finally {
      setDetailLoading(false);
    }
  };

  if (loading) return <LoadingScreen />;
  if (error && !snapshot) return <ErrorScreen error={error} onRetry={loadSnapshot} />;
  if (!snapshot) return null;

  const summary = snapshot.summary;
  const topTopic = snapshot.topics[0];
  const topChannel = snapshot.channels[0];
  const topVideo = snapshot.videos[0];
  const timeline = snapshot.timeseries.map((item) => ({ ...item, month: formatDate(item.published_month) }));
  const topicTimeline = buildTopicTimeline(snapshot);
  const currentRows = videoData?.rows || snapshot.videos;
  const totalPages = Math.max(1, Math.ceil((videoData?.total || snapshot.videos.length) / 50));
  const topicCount = snapshot.topics.reduce((sum, item) => sum + item.videos, 0);

  return <div className="theme-root min-h-screen overflow-x-hidden text-slate-100">
    <header className="dashboard-header sticky top-0 z-30 border-b border-white/10 bg-[#07131f]/85 backdrop-blur-xl"><div className="mx-auto flex max-w-[1480px] items-center justify-between px-5 py-4 lg:px-8"><div className="flex items-center gap-3"><button className="rounded-lg p-2 text-slate-400 hover:bg-white/10 lg:hidden" onClick={() => setMobileMenu(!mobileMenu)}><Menu size={18} /></button><div className="brand-mark grid h-10 w-10 place-items-center rounded-xl bg-gradient-to-br from-blue-400 to-teal-300 text-[#07131f] shadow-lg shadow-cyan-500/20"><Sparkles size={19} /></div><div><p className="text-sm font-extrabold tracking-tight text-white">Signal Garden</p><p className="mono text-[10px] uppercase tracking-[0.18em] text-slate-500">YouTube research lab</p></div></div><div className="hidden items-center gap-2 md:flex"><span className={cn("status-dot h-2 w-2 rounded-full", live ? "bg-emerald-400 shadow-[0_0_12px_#34d399]" : "bg-rose-400")} /><span className="mono text-[10px] uppercase tracking-[0.16em] text-slate-400">{live ? "live duckdb" : "api offline"}</span><span className="mx-2 h-5 w-px bg-white/10" /><span className="mono text-xs text-slate-500">{snapshot.run.id}</span></div></div></header>
    <main className="dashboard-main mx-auto max-w-[1480px] px-5 pb-20 pt-8 lg:px-8">
      <section className="dashboard-hero grid-fade relative overflow-hidden rounded-[2rem] border border-cyan-300/15 bg-gradient-to-br from-[#102f46] via-[#0c2233] to-[#0b1d2c] p-7 shadow-2xl shadow-black/20 lg:p-10"><div className="absolute -right-24 -top-36 h-96 w-96 rounded-full bg-cyan-400/10 blur-3xl" /><div className="relative grid gap-8 lg:grid-cols-[1.35fr_0.65fr] lg:items-end"><div><div className="mb-4 inline-flex items-center gap-2 rounded-full border border-cyan-200/20 bg-cyan-200/10 px-3 py-1.5 text-[11px] font-bold text-cyan-200"><Activity size={13} />{labelFor(snapshot.run.status)} · {snapshot.run.language || "language unknown"}</div><h1 className="max-w-4xl text-4xl font-extrabold leading-[1.05] tracking-[-0.04em] text-white md:text-6xl">{snapshot.run.topicQuery || "Untitled research run"}</h1><p className="mt-5 max-w-2xl text-sm leading-7 text-slate-300">A keyword-first view of what this crawl found, what made it through transcript review, and how the conversation changes over time.</p><div className="mt-6 flex flex-wrap gap-2"><Badge color="#67e8f9">start {snapshot.run.startDate || "unknown"}</Badge><Badge color="#a7f3d0">{formatDate(snapshot.run.firstPublished)} → {formatDate(snapshot.run.lastPublished)}</Badge><Badge color="#fde68a">{snapshot.run.expandedQueries.length} planned query variants</Badge><Badge color="#c4b5fd">topics overlap by design</Badge></div></div><div className="quick-read surface-soft rounded-2xl p-5"><p className="mono text-[10px] uppercase tracking-[0.18em] text-cyan-200/70">the quick read</p><p className="mt-3 text-sm leading-6 text-slate-200"><span className="font-bold text-white">{topTopic?.label || "The topic map"}</span> is the most common theme in date-eligible videos. <span className="font-bold text-white">{topChannel?.channel_title || "One channel"}</span> has the largest observed view total, while <span className="font-bold text-white">{topVideo?.title || "the leading video"}</span> is the biggest single view outlier.</p><p className="mt-4 text-xs leading-5 text-slate-400">These are descriptive signals, not causal or representative estimates. Views and likes are the values captured by SearchAPI at crawl time.</p></div></div></section>
      <section className="mt-5 grid gap-4 sm:grid-cols-2 xl:grid-cols-6"><MetricCard label="Candidates" value={formatNumber(summary.candidates, 0)} note={formatPercent(summary.acceptedRate) + " final accepted"} accent="#60a5fa" icon={<Layers3 size={18} />} /><MetricCard label="Accepted" value={formatNumber(summary.accepted, 0)} note={formatNumber(summary.transcripts_available, 0) + " transcripts available"} accent="#34d399" icon={<Check size={18} />} /><MetricCard label="Observed views" value={formatNumber(summary.total_views)} note={formatNumber(summary.median_views, 0) + " median / video"} accent="#fbbf24" icon={<Play size={18} />} /><MetricCard label="Observed likes" value={formatNumber(summary.total_likes)} note={formatNumber(summary.median_likes, 0) + " median / video"} accent="#fb7185" icon={<Sparkles size={18} />} /><MetricCard label="Channels" value={formatNumber(summary.channels, 0)} note={formatNumber(topChannel?.videos, 0) + " videos in top channel"} accent="#c084fc" icon={<BarChart3 size={18} />} /><MetricCard label="Transcript words" value={formatNumber(summary.transcript_words)} note={formatNumber(summary.transcript_segments, 0) + " segments"} accent="#2dd4bf" icon={<Table2 size={18} />} /></section>
      <section className="mt-12"><SectionHeading eyebrow="01 · the shape of the corpus" title="Discovery, acceptance, and time" detail="Candidate volume is not the same as research evidence. The funnel keeps those states visible." icon={<Activity size={13} />} /><div className="grid gap-5 xl:grid-cols-[1.15fr_0.85fr]"><Panel><div className="mb-4 flex items-center justify-between"><div><h3 className="font-bold text-white">Publication rhythm</h3><p className="mt-1 text-xs text-slate-400">Date-eligible videos by publication month</p></div><Badge color="#60a5fa">videos · accepted · transcripts</Badge></div><div className="h-72"><ResponsiveContainer width="100%" height="100%"><ComposedChart data={timeline} margin={{ top: 10, right: 4, left: -12, bottom: 0 }}><CartesianGrid stroke="rgba(148,163,184,.12)" vertical={false} /><XAxis dataKey="month" tick={{ fill: "#91a6b4", fontSize: 10 }} tickLine={false} axisLine={false} minTickGap={28} /><YAxis tick={{ fill: "#91a6b4", fontSize: 10 }} tickLine={false} axisLine={false} /><Tooltip content={<ChartTooltip />} /><Area type="monotone" dataKey="videos" name="videos" fill="#60a5fa" fillOpacity={0.16} stroke="#60a5fa" strokeWidth={2} /><Line type="monotone" dataKey="accepted" name="accepted" stroke="#34d399" strokeWidth={2.5} dot={false} /><Line type="monotone" dataKey="transcribed" name="transcribed" stroke="#fbbf24" strokeWidth={1.8} dot={false} /></ComposedChart></ResponsiveContainer></div></Panel><Panel><div className="mb-4 flex items-center justify-between"><div><h3 className="font-bold text-white">Funnel to evidence</h3><p className="mt-1 text-xs text-slate-400">Only transcript-stage relevant is accepted.</p></div><MousePointerClick size={17} className="text-cyan-300" /></div><div className="space-y-3">{snapshot.funnel.map((item, index) => <div key={item.label}><div className="mb-1.5 flex justify-between text-xs"><span className="font-semibold text-slate-300">{item.label}</span><span className="mono text-slate-400">{formatNumber(item.count, 0)} · {item.percent.toFixed(1)}%</span></div><div className="h-3 overflow-hidden rounded-full bg-slate-800"><div className="h-full rounded-full" style={{ width: Math.max(item.percent, item.count ? 1 : 0) + "%", background: palette[index] }} /></div></div>)}</div><div className="mt-6 grid grid-cols-2 gap-3"><MiniStat label="date start" value={snapshot.run.startDate || "—"} /><MiniStat label="coverage" value={formatPercent(summary.transcriptCoverage)} /><MiniStat label="with views" value={formatPercent(summary.videos_with_views / Math.max(summary.candidates, 1))} /><MiniStat label="with likes" value={formatPercent(summary.videos_with_likes / Math.max(summary.candidates, 1))} /></div></Panel></div></section>
      <section className="mt-12"><SectionHeading eyebrow="02 · topic model" title="A manual map of the conversation" detail={"Eight transparent keyword themes. " + formatNumber(topicCount, 0) + " topic hits across " + formatNumber(summary.candidates, 0) + " candidates means overlap is expected."} icon={<Sparkles size={13} />} /><div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">{snapshot.topics.map((topic) => <TopicCard key={topic.topic_id} topic={topic} total={Math.max(summary.candidates, 1)} active={activeTopic === topic.topic_id} onClick={() => { setActiveTopic(activeTopic === topic.topic_id ? "" : topic.topic_id); setPage(0); document.getElementById("video-explorer")?.scrollIntoView({ behavior: "smooth", block: "start" }); }} />)}</div></section>
      <section className="mt-5 grid gap-5 xl:grid-cols-[1.1fr_0.9fr]"><Panel><SectionHeading eyebrow="topic over time" title="Which themes gained ground?" detail="The top five themes are shown as monthly hit counts." /><div className="h-80"><ResponsiveContainer width="100%" height="100%"><LineChart data={topicTimeline} margin={{ top: 6, right: 8, left: -12, bottom: 0 }}><CartesianGrid stroke="rgba(148,163,184,.12)" vertical={false} /><XAxis dataKey="month" tick={{ fill: "#91a6b4", fontSize: 10 }} tickLine={false} axisLine={false} minTickGap={28} /><YAxis tick={{ fill: "#91a6b4", fontSize: 10 }} tickLine={false} axisLine={false} /><Tooltip content={<ChartTooltip />} />{snapshot.topics.slice(0, 5).map((topic) => <Line key={topic.topic_id} type="monotone" dataKey={topic.topic_id} name={topic.label} stroke={topic.color} strokeWidth={2} dot={false} />)}<Legend wrapperStyle={{ fontSize: 10, color: "#9fb4c0" }} /></LineChart></ResponsiveContainer></div></Panel><Panel><SectionHeading eyebrow="source mix" title="How videos entered the corpus" detail="Discovery provenance, not channel quality." /><div className="h-80"><ResponsiveContainer width="100%" height="100%"><PieChart><Pie data={snapshot.sourceMix} dataKey="videos" nameKey="source" cx="50%" cy="48%" innerRadius={68} outerRadius={105} paddingAngle={3} stroke="none">{snapshot.sourceMix.map((entry, index) => <Cell key={entry.source} fill={palette[index % palette.length]} />)}</Pie><Tooltip content={<ChartTooltip />} /><Legend wrapperStyle={{ fontSize: 11, color: "#9fb4c0" }} /></PieChart></ResponsiveContainer></div></Panel></section>
      <section className="mt-12"><SectionHeading eyebrow="03 · attention & reach" title="Engagement is long-tailed" detail="The median is more honest than the mean here. Hover the scatter to inspect view outliers." icon={<BarChart3 size={13} />} /><div className="grid gap-5 xl:grid-cols-[1.15fr_0.85fr]"><Panel><div className="mb-4 flex items-center justify-between"><div><h3 className="font-bold text-white">Views × likes</h3><p className="mt-1 text-xs text-slate-400">Top 80 by observed views</p></div><Badge color="#fbbf24">descriptive only</Badge></div><div className="h-80"><ResponsiveContainer width="100%" height="100%"><ScatterChart margin={{ top: 10, right: 16, left: -8, bottom: 8 }}><CartesianGrid stroke="rgba(148,163,184,.12)" /><XAxis type="number" dataKey="views" name="views" tick={{ fill: "#91a6b4", fontSize: 10 }} tickLine={false} axisLine={false} tickFormatter={(value) => formatNumber(value, 0)} /><YAxis type="number" dataKey="likes" name="likes" tick={{ fill: "#91a6b4", fontSize: 10 }} tickLine={false} axisLine={false} tickFormatter={(value) => formatNumber(value, 0)} /><Tooltip cursor={{ strokeDasharray: "4 4" }} content={<ChartTooltip />} /><Scatter data={snapshot.videos.filter((item) => item.views !== null && item.likes !== null)} fill="#60a5fa" fillOpacity={0.6} /></ScatterChart></ResponsiveContainer></div></Panel><Panel><SectionHeading eyebrow="metadata quality" title="What is actually measurable?" detail="Availability is a property of the capture, not of YouTube in general." /><div className="space-y-4">{Object.entries(snapshot.metadata.coverage).map(([key, value], index) => <div key={key}><div className="mb-1.5 flex justify-between text-xs"><span className="capitalize text-slate-300">{key.replace(/([A-Z])/g, " $1")}</span><span className="mono text-slate-400">{Number(value).toFixed(0)}%</span></div><div className="h-2 rounded-full bg-slate-800"><div className="h-full rounded-full" style={{ width: Number(value) + "%", background: palette[index % palette.length] }} /></div></div>)}</div><div className="mt-7 rounded-xl border border-amber-200/15 bg-amber-300/5 p-4 text-xs leading-5 text-amber-100/75">Views and likes are not exposure-normalized. A channel's age, audience, upload cadence, and snapshot date all matter.</div></Panel></div></section>
      <section className="mt-12"><SectionHeading eyebrow="04 · channels & language" title="The channel landscape" detail="Aggregated from candidates and joined to captured subscriber metadata." icon={<Layers3 size={13} />} /><div className="grid gap-5 xl:grid-cols-[1.35fr_0.65fr]"><Panel className="overflow-hidden"><div className="overflow-x-auto"><table className="w-full min-w-[720px] text-left text-xs"><thead className="border-b border-white/10 text-[10px] uppercase tracking-[0.12em] text-slate-500"><tr><th className="px-4 py-3">channel</th><th className="px-4 py-3 text-right">videos</th><th className="px-4 py-3 text-right">accepted</th><th className="px-4 py-3 text-right">views</th><th className="px-4 py-3 text-right">median views</th><th className="px-4 py-3 text-right">subscribers</th></tr></thead><tbody className="divide-y divide-white/5">{snapshot.channels.slice(0, 12).map((item, index) => <tr key={item.channel_id} className="transition hover:bg-white/[0.04]"><td className="max-w-[260px] px-4 py-3"><button className="truncate font-semibold text-slate-200 hover:text-cyan-300" onClick={() => { setChannel(item.channel_id); setPage(0); document.getElementById("video-explorer")?.scrollIntoView({ behavior: "smooth" }); }}>{index + 1}. {item.channel_title}</button><div className="mono mt-1 truncate text-[10px] text-slate-600">{item.channel_id}</div></td><td className="px-4 py-3 text-right mono text-slate-300">{formatNumber(item.videos, 0)}</td><td className="px-4 py-3 text-right mono text-emerald-300">{formatNumber(item.accepted, 0)}</td><td className="px-4 py-3 text-right mono text-amber-200">{formatNumber(item.views)}</td><td className="px-4 py-3 text-right mono text-slate-300">{formatNumber(item.median_views, 0)}</td><td className="px-4 py-3 text-right mono text-slate-400">{formatNumber(item.subscribers, 0)}</td></tr>)}</tbody></table></div></Panel><Panel><SectionHeading eyebrow="word explorer" title="Frequent title & description terms" detail="Stopwords and URLs are removed in DuckDB. Transcript search remains live." /><div className="flex flex-wrap gap-2">{snapshot.keywords.slice(0, 34).map((item) => <button key={item.word} onClick={() => { setSearch(item.word); setPage(0); document.getElementById("video-explorer")?.scrollIntoView({ behavior: "smooth" }); }} className="group rounded-xl border border-white/10 bg-white/[0.03] px-3 py-2 text-left transition hover:border-cyan-300/40 hover:bg-cyan-300/10"><span className="block text-xs font-bold text-slate-200 group-hover:text-cyan-200">{item.word}</span><span className="mono mt-1 block text-[10px] text-slate-500">{formatNumber(item.videos, 0)} videos · {formatNumber(item.mentions, 0)} hits</span></button>)}</div></Panel></div></section>
      <section id="video-explorer" className="mt-12 scroll-mt-24"><SectionHeading eyebrow="05 · drill-down" title="Explore every candidate" detail="Server-side DuckDB filtering keeps the browser light. Click any row for its transcript and keyword evidence." icon={<MousePointerClick size={13} />} /><Panel className="overflow-hidden"><div className="flex flex-col gap-3 border-b border-white/10 p-4 lg:flex-row lg:items-center"><label className="relative flex-1"><Search size={15} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" /><input value={search} onChange={(event) => { setSearch(event.target.value); setPage(0); }} placeholder="Search title, description, keywords, or transcript…" className="w-full rounded-xl border border-white/10 bg-black/10 py-2.5 pl-9 pr-3 text-sm text-slate-200 outline-none placeholder:text-slate-600 focus:border-cyan-300/50" /></label><div className="flex flex-wrap gap-2"><Select value={activeTopic} onChange={(value) => { setActiveTopic(value); setPage(0); }} options={[["", "All topics"], ...snapshot.topicDefinitions.map((item) => [item.id, item.label] as [string, string])]} /><Select value={label} onChange={(value) => { setLabel(value); setPage(0); }} options={[["", "All states"], ["accepted", "Accepted"], ["irrelevant", "Out of scope"], ["needs_transcript", "Needs transcript"], ["deferred_budget", "Budget deferred"]]} /><Select value={channel} onChange={(value) => { setChannel(value); setPage(0); }} options={[["", "All channels"], ...snapshot.channels.map((item) => [item.channel_id, item.channel_title] as [string, string])]} /><Select value={sort} onChange={(value) => { setSort(value); setPage(0); }} options={[["views", "Sort: views"], ["likes", "Sort: likes"], ["published", "Sort: newest"], ["duration", "Sort: duration"]]} /></div></div>{(activeTopic || label || channel || search) && <div className="flex items-center gap-2 border-b border-white/5 bg-cyan-300/[0.03] px-4 py-2.5 text-xs text-cyan-100/75"><Filter size={13} /><span>{videoData?.total || 0} rows match these filters</span><button className="ml-auto flex items-center gap-1 text-slate-400 hover:text-white" onClick={() => { setActiveTopic(""); setLabel(""); setChannel(""); setSearch(""); setPage(0); }}><X size={13} /> clear</button></div>}<VideoTable rows={currentRows} loading={videoLoading} onOpen={openVideo} /><div className="flex items-center justify-between border-t border-white/10 px-4 py-3 text-xs text-slate-500"><span className="mono">{videoData?.total || snapshot.videos.length} results · page {page + 1} / {totalPages}</span><div className="flex gap-1"><button disabled={page === 0} onClick={() => setPage((value) => Math.max(0, value - 1))} className="rounded-lg border border-white/10 p-2 disabled:opacity-30"><ChevronLeft size={15} /></button><button disabled={page + 1 >= totalPages} onClick={() => setPage((value) => Math.min(totalPages - 1, value + 1))} className="rounded-lg border border-white/10 p-2 disabled:opacity-30"><ChevronRight size={15} /></button></div></div></Panel></section>
      <section className="mt-12 grid gap-5 xl:grid-cols-[0.9fr_1.1fr]"><Panel><SectionHeading eyebrow="query plan" title="What the crawler asked for" detail="Query strings are preserved as research provenance." /><div className="space-y-2">{snapshot.queries.slice(0, 12).map((item) => <div key={item.query + item.status} className="flex items-center gap-3 rounded-xl border border-white/5 bg-black/10 p-3"><span className="grid h-7 w-7 shrink-0 place-items-center rounded-lg bg-blue-400/10 text-blue-300"><Search size={13} /></span><div className="min-w-0 flex-1"><p className="truncate text-xs font-semibold text-slate-300">{item.query}</p><p className="mono mt-1 text-[10px] text-slate-600">{item.kind} · {item.status}</p></div><div className="text-right"><p className="mono text-xs text-slate-300">{formatNumber(item.videos, 0)}</p><p className="text-[10px] text-emerald-300">{formatNumber(item.accepted, 0)} accepted</p></div></div>)}</div></Panel><Panel><SectionHeading eyebrow="interpretation guardrails" title="What this dashboard can and cannot say" detail="Every aggregate points back to a JSONL record." /><div className="grid gap-3 sm:grid-cols-2"><Guardrail title="Good for" items={["keyword-defined topic prevalence", "channel and source comparisons", "publication timing and volume", "descriptive views / likes distributions", "transcript coverage and searchable evidence"]} color="#34d399" /><Guardrail title="Do not over-read" items={["causal influence or audience effects", "YouTube-wide representativeness", "topic exclusivity", "current live engagement", "semantic meaning beyond keywords"]} color="#fb7185" /></div><p className="mt-5 text-xs leading-6 text-slate-500">The topic model is manual: it was derived from the run specification, interview examples, and observed wording. Change the definitions in <span className="mono text-slate-400">src/yt_searchapi/analysis.py</span> when the research question changes.</p></Panel></section>
    </main>
    {(selectedVideo || detailLoading) && <DetailDrawer video={selectedVideo} loading={detailLoading} onClose={() => setSelectedVideo(null)} />}
    {error && snapshot && <div className="fixed bottom-5 left-1/2 z-50 flex -translate-x-1/2 items-center gap-3 rounded-xl border border-rose-300/20 bg-rose-950/90 px-4 py-3 text-xs text-rose-100 shadow-2xl"><CircleHelp size={15} /><span>{error}</span><button onClick={() => setError(null)}><X size={14} /></button></div>}
    {mobileMenu && <div className="fixed inset-0 z-20 bg-black/40 lg:hidden" onClick={() => setMobileMenu(false)} />}
  </div>;
}

function LoadingScreen() {
  return <div className="loading-screen grid min-h-screen place-items-center bg-[#07131f]"><div className="text-center"><div className="mx-auto mb-4 h-10 w-10 animate-spin rounded-full border-2 border-cyan-300/20 border-t-cyan-300" /><p className="mono text-xs uppercase tracking-[0.2em] text-slate-500">building duckdb signal map</p></div></div>;
}

function ErrorScreen({ error, onRetry }: { error: string; onRetry: () => void }) {
  return <div className="error-screen grid min-h-screen place-items-center bg-[#07131f] p-6"><div className="surface max-w-lg rounded-2xl p-7"><p className="mono text-xs uppercase tracking-[0.2em] text-rose-300">dashboard unavailable</p><h1 className="mt-3 text-2xl font-extrabold text-white">Start the live query layer first.</h1><p className="mt-3 text-sm leading-6 text-slate-400">{error}</p><button onClick={onRetry} className="mt-6 rounded-xl bg-cyan-300 px-4 py-2 text-sm font-bold text-[#07131f]">Retry</button></div></div>;
}

function TopicCard({ topic, total, active, onClick }: { topic: Topic; total: number; active: boolean; onClick: () => void }) {
  const keywords = [...new Set(topic.matched_keywords?.flat() || [])];
  return <button onClick={onClick} className={cn("topic-card group relative overflow-hidden rounded-2xl border p-5 text-left transition", active ? "border-cyan-200/60 bg-cyan-200/10" : "border-white/10 bg-[#0b1d2c]/80 hover:-translate-y-0.5 hover:border-white/25")}><div className="absolute right-0 top-0 h-24 w-24 rounded-full opacity-15 blur-2xl" style={{ background: topic.color }} /><div className="relative"><div className="flex items-start justify-between gap-3"><span className="h-2.5 w-2.5 rounded-full" style={{ background: topic.color, boxShadow: "0 0 14px " + topic.color }} /><span className="mono text-[10px] text-slate-500">{formatNumber(topic.videos, 0)} hits</span></div><h3 className="mt-5 text-lg font-extrabold text-white">{topic.label}</h3><p className="mt-2 min-h-10 text-xs leading-5 text-slate-400">{topic.description}</p><div className="mt-4 h-1.5 rounded-full bg-slate-800"><div className="h-full rounded-full" style={{ width: Math.min(100, topic.videos / total * 100) + "%", background: topic.color }} /></div><div className="mt-4 flex items-center justify-between text-xs"><span className="text-emerald-300">{formatNumber(topic.accepted, 0)} accepted</span><span className="mono text-slate-400">{formatNumber(topic.median_views, 0)} median views</span></div><div className="mt-3 flex flex-wrap gap-1.5">{keywords.slice(0, 4).map((keyword) => <span key={keyword} className="keyword-token">{keyword}</span>)}</div></div></button>;
}

function buildTopicTimeline(snapshot: Snapshot) {
  const ids = snapshot.topics.slice(0, 5).map((item) => item.topic_id);
  const byMonth = new Map<string, Record<string, string | number>>();
  snapshot.topicTimeseries.forEach((row) => {
    if (!ids.includes(row.topic_id)) return;
    const month = formatDate(row.published_month);
    const current = byMonth.get(month) || { month };
    current[row.topic_id] = row.videos;
    byMonth.set(month, current);
  });
  return [...byMonth.values()];
}

function Select({ value, onChange, options }: { value: string; onChange: (value: string) => void; options: [string, string][] }) {
  return <select value={value} onChange={(event) => onChange(event.target.value)} className="theme-select max-w-[190px] rounded-xl border border-white/10 bg-[#102b3e] px-3 py-2.5 text-xs font-semibold text-slate-300 outline-none focus:border-cyan-300/50">{options.map(([optionValue, label]) => <option key={optionValue} value={optionValue}>{label}</option>)}</select>;
}

function VideoTable({ rows, loading, onOpen }: { rows: VideoRow[]; loading: boolean; onOpen: (id: string) => void }) {
  const columns = useMemo<ColumnDef<VideoRow>[]>(() => [
    { accessorKey: "title", header: "video", cell: ({ row }) => <button className="max-w-[390px] text-left" onClick={() => onOpen(row.original.video_id)}><p className="line-clamp-2 text-xs font-bold text-slate-200 hover:text-cyan-300">{row.original.title}</p><p className="mt-1 truncate text-[10px] text-slate-500">{row.original.channel_title || "Unknown channel"} · {formatDate(row.original.published_at)}</p></button> },
    { accessorKey: "final_label", header: "state", cell: ({ row }) => <Badge color={row.original.final_label === "relevant" ? "#34d399" : row.original.final_label === "irrelevant" ? "#fb7185" : "#fbbf24"}>{labelFor(row.original.final_label)}</Badge> },
    { accessorKey: "topics", header: "topics", cell: ({ row }) => <div className="flex max-w-[220px] flex-wrap gap-1">{row.original.topics?.filter(Boolean).slice(0, 2).map((topic) => <span key={topic} className="topic-pill">{topic}</span>)}</div> },
    { accessorKey: "views", header: "views", cell: ({ row }) => <span className="mono text-xs text-amber-200">{formatNumber(row.original.views)}</span> },
    { accessorKey: "likes", header: "likes", cell: ({ row }) => <span className="mono text-xs text-rose-200">{formatNumber(row.original.likes)}</span> },
    { accessorKey: "transcript_words", header: "transcript", cell: ({ row }) => <span className="mono text-xs text-slate-400">{row.original.transcript_available ? formatNumber(row.original.transcript_words, 0) : "—"}</span> },
  ], [onOpen]);
  const table = useReactTable({ data: rows, columns, getCoreRowModel: getCoreRowModel() });
  return <div className="video-table relative overflow-x-auto">{loading && <div className="absolute inset-x-0 top-0 z-10 h-0.5 animate-pulse bg-cyan-300" />}<table className="w-full min-w-[970px] text-left"><thead className="border-b border-white/10 bg-black/10 text-[10px] uppercase tracking-[0.12em] text-slate-500"><tr>{table.getHeaderGroups().map((group) => group.headers.map((header) => <th key={header.id} className="px-4 py-3 font-semibold">{flexRender(header.column.columnDef.header, header.getContext())}</th>))}</tr></thead><tbody className="divide-y divide-white/5">{table.getRowModel().rows.map((row) => <tr key={row.id} className="transition hover:bg-white/[0.04]">{row.getVisibleCells().map((cell) => <td key={cell.id} className="px-4 py-3 align-top">{flexRender(cell.column.columnDef.cell, cell.getContext())}</td>)}</tr>)}</tbody></table>{!rows.length && <div className="p-12 text-center text-sm text-slate-500">No rows match these filters.</div>}</div>;
}

function Guardrail({ title, items, color }: { title: string; items: string[]; color: string }) {
  return <div className="guardrail rounded-2xl border p-4" style={{ borderColor: color + "28", background: color + "08" }}><div className="mb-3 flex items-center gap-2 text-sm font-bold" style={{ color }}><span className="h-2 w-2 rounded-full" style={{ background: color }} />{title}</div><ul className="space-y-2 text-xs leading-5 text-slate-400">{items.map((item) => <li key={item} className="flex gap-2"><span className="text-slate-600">•</span>{item}</li>)}</ul></div>;
}

function DetailDrawer({ video, loading, onClose }: { video: VideoDetail | null; loading: boolean; onClose: () => void }) {
  return <div className="detail-drawer fixed inset-0 z-50 flex justify-end bg-black/50 backdrop-blur-sm" onClick={onClose}><aside className="detail-drawer__panel h-full w-full max-w-2xl overflow-y-auto border-l border-white/10 bg-[#091a29] p-6 shadow-2xl" onClick={(event) => event.stopPropagation()}>{loading && !video ? <div className="grid h-full place-items-center"><div className="h-8 w-8 animate-spin rounded-full border-2 border-cyan-300/20 border-t-cyan-300" /></div> : video ? <><div className="flex items-start justify-between gap-4"><div><p className="mono text-[10px] uppercase tracking-[0.2em] text-cyan-300/70">video evidence</p><h2 className="mt-3 text-2xl font-extrabold leading-tight text-white">{video.title}</h2><p className="mt-2 text-xs text-slate-500">{video.channel_title} · {formatDate(video.published_at)}</p></div><button onClick={onClose} className="rounded-lg p-2 text-slate-400 hover:bg-white/10 hover:text-white"><X size={18} /></button></div><div className="mt-6 flex flex-wrap gap-2"><Badge color="#fbbf24">{formatNumber(video.views)} views</Badge><Badge color="#fb7185">{formatNumber(video.likes)} likes</Badge><Badge color="#60a5fa">{formatDuration(video.duration_seconds)}</Badge><Badge color={video.final_label === "relevant" ? "#34d399" : "#fb7185"}>{labelFor(video.final_label)}</Badge>{video.topics?.filter(Boolean).map((topic) => <Badge key={topic} color="#67e8f9">{topic}</Badge>)}</div><div className="mt-7 grid grid-cols-2 gap-3"><MiniStat label="discovery" value={video.discovered_via || "—"} /><MiniStat label="primary reason" value={video.primary_reason || "—"} /><MiniStat label="transcript words" value={formatNumber(video.transcript_words, 0)} /><MiniStat label="segments" value={formatNumber(video.transcript_segments.length, 0)} /></div><div className="mt-7"><p className="mono mb-2 text-[10px] uppercase tracking-[0.16em] text-cyan-300/70">description</p><p className="whitespace-pre-wrap text-sm leading-7 text-slate-300">{video.description || "No description captured."}</p></div><div className="mt-7"><div className="mb-2 flex items-center justify-between"><p className="mono text-[10px] uppercase tracking-[0.16em] text-cyan-300/70">transcript</p>{video.url && <a href={video.url} target="_blank" rel="noreferrer" className="flex items-center gap-1 text-xs text-cyan-300 hover:text-white">open on YouTube <ExternalLink size={12} /></a>}</div><div className="max-h-[30rem] overflow-y-auto rounded-xl border border-white/10 bg-black/15 p-4 text-sm leading-7 text-slate-300">{video.transcript_text || "No transcript text is available for this row."}</div></div><div className="mt-7"><p className="mono mb-2 text-[10px] uppercase tracking-[0.16em] text-cyan-300/70">matched keywords</p><div className="flex flex-wrap gap-2">{video.matched_topic_keywords?.flat().map((keyword) => <span key={keyword} className="keyword-token" >{keyword}</span>)}</div></div></> : null}</aside></div>;
}
