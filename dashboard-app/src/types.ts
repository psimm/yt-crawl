export type Topic = {
  topic_id: string;
  label: string;
  color: string;
  description: string;
  videos: number;
  accepted: number;
  views: number;
  likes: number;
  median_views: number | null;
  average_likes: number | null;
  transcribed: number;
  matched_keywords: string[][];
};

export type TopicDefinition = {
  id: string;
  label: string;
  color: string;
  description: string;
  keywords: string[];
};

export type VideoRow = {
  video_id: string;
  title: string;
  description: string | null;
  url: string | null;
  channel_id: string | null;
  channel_title: string | null;
  published_at: string | null;
  published_month: string | null;
  duration_seconds: number | null;
  views: number | null;
  likes: number | null;
  category: string | null;
  keywords: string[] | null;
  thumbnail: string | null;
  discovered_via: string | null;
  discovery_query: string | null;
  final_label: string | null;
  decision_point: string | null;
  confidence: number | null;
  reason: string | null;
  transcript_available: boolean;
  transcript_words: number;
  transcript_segments: number;
  topics: string[];
  topic_ids: string[];
};

export type Channel = {
  channel_id: string;
  channel_title: string;
  subscribers: number | null;
  channel_views: number | null;
  videos: number;
  accepted: number;
  views: number;
  likes: number;
  median_views: number | null;
  average_likes: number | null;
  transcribed: number;
  topic_count: number;
};

export type Snapshot = {
  run: {
    id: string;
    topicQuery: string | null;
    expandedQueries: string[];
    language: string | null;
    startDate: string | null;
    model: string | null;
    promptVersion: string | null;
    status: string;
    statusReason: string | null;
    startedAt: string | null;
    firstPublished: string | null;
    lastPublished: string | null;
  };
  summary: {
    candidates: number;
    accepted: number;
    irrelevant: number;
    pending: number;
    channels: number;
    videos_with_views: number;
    videos_with_likes: number;
    total_views: number;
    total_likes: number;
    median_views: number | null;
    median_likes: number | null;
    average_views: number | null;
    average_likes: number | null;
    transcripts_available: number;
    transcripts_seen: number;
    transcript_words: number;
    transcript_segments: number;
    transcriptCoverage: number | null;
    acceptedRate: number | null;
    viewsPerAccepted: number | null;
  };
  funnel: { label: string; count: number; percent: number }[];
  timeseries: {
    published_month: string;
    videos: number;
    accepted: number;
    transcribed: number;
    views: number;
    median_views: number | null;
    median_likes: number | null;
  }[];
  topicTimeseries: {
    published_month: string;
    topic_id: string;
    label: string;
    color: string;
    videos: number;
    accepted: number;
  }[];
  topics: Topic[];
  channels: Channel[];
  keywords: { word: string; mentions: number; videos: number }[];
  sourceMix: { source: string; videos: number; accepted: number; views: number }[];
  metadata: Record<string, unknown> & {
    coverage: Record<string, number>;
  };
  queries: {
    query: string;
    kind: string;
    status: string;
    videos: number;
    accepted: number;
  }[];
  videos: VideoRow[];
  topicDefinitions: TopicDefinition[];
  server: { fingerprint: string; recordTypes: string[] };
};

export type VideoDetail = Omit<VideoRow, "transcript_segments"> & {
  published_date: string | null;
  in_scope_date: boolean | null;
  transcript_text: string;
  search_text: string;
  topic_ids: string[];
  matched_topic_keywords: string[][] | null;
  transcript_segments: {
    segment_index: number;
    text: string;
    start_seconds: number;
    duration_seconds: number;
  }[];
  fingerprint: string;
};
