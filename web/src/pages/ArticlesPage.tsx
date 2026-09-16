import { useCallback, useEffect, useState } from "react";
import {
  BookMarked,
  BookOpen,
  Check,
  Download,
  ExternalLink,
  X,
} from "lucide-react";
import { Button } from "@nous-research/ui/ui/components/button";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { H2 } from "@nous-research/ui/ui/components/typography/h2";
import { Card, CardContent } from "@nous-research/ui/ui/components/card";
import { useToast } from "@nous-research/ui/hooks/use-toast";
import { Toast } from "@nous-research/ui/ui/components/toast";
import { usePageHeader } from "@/contexts/usePageHeader";
import { api } from "@/lib/api";
import type { Article, ArticleStatus } from "@/lib/api";

const TABS: { key: ArticleStatus | "all"; label: string }[] = [
  { key: "new", label: "New" },
  { key: "saved", label: "Saved" },
  { key: "read", label: "Read" },
  { key: "all", label: "All" },
];

export default function ArticlesPage() {
  const [status, setStatus] = useState<ArticleStatus | "all">("new");
  const [articles, setArticles] = useState<Article[]>([]);
  const [keywords, setKeywords] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [fetching, setFetching] = useState(false);
  const [updating, setUpdating] = useState<string | null>(null);
  const { toast, showToast } = useToast();
  const { setEnd } = usePageHeader();

  const load = useCallback(
    (s: ArticleStatus | "all") => {
      setLoading(true);
      return api
        .getArticles(s)
        .then((res) => {
          setArticles(res.articles);
          setKeywords(res.keywords);
        })
        .catch(() => showToast("Failed to load articles", "error"))
        .finally(() => setLoading(false));
    },
    [showToast],
  );

  useEffect(() => {
    load(status);
  }, [status, load]);

  const handleFetch = async () => {
    setFetching(true);
    try {
      const res = await api.fetchArticles();
      showToast(`Fetched ${res.new_articles} new article(s)`, "success");
      load(status);
    } catch (e) {
      showToast(`Error: ${e}`, "error");
    } finally {
      setFetching(false);
    }
  };

  const handleSetStatus = async (id: string, next: ArticleStatus) => {
    setUpdating(id);
    try {
      await api.updateArticleStatus(id, next);
      // Optimistic: an article leaves the current tab once its status changes.
      setArticles((prev) => prev.filter((a) => a.id !== id));
    } catch (e) {
      showToast(`Error: ${e}`, "error");
    } finally {
      setUpdating(null);
    }
  };

  useEffect(() => {
    setEnd(
      <Button
        className="uppercase"
        size="sm"
        onClick={handleFetch}
        disabled={fetching}
        prefix={fetching ? <Spinner /> : <Download className="h-4 w-4" />}
      >
        Fetch new
      </Button>,
    );
    return () => setEnd(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [setEnd, fetching]);

  return (
    <div className="flex flex-col gap-6">
      <Toast toast={toast} />

      <div className="flex items-center justify-between">
        <H2 variant="sm" className="flex items-center gap-2 text-muted-foreground">
          <BookOpen className="h-4 w-4" />
          PhD reading list
        </H2>
        {keywords.length > 0 && (
          <div className="flex flex-wrap items-center gap-1.5">
            {keywords.map((k) => (
              <Badge key={k} tone="outline">
                {k}
              </Badge>
            ))}
          </div>
        )}
      </div>

      <div className="flex items-center gap-1">
        {TABS.map((tab) => (
          <Button
            key={tab.key}
            size="sm"
            ghost={status !== tab.key}
            className="uppercase"
            onClick={() => setStatus(tab.key)}
          >
            {tab.label}
          </Button>
        ))}
      </div>

      {loading ? (
        <div className="flex items-center justify-center py-24">
          <Spinner className="text-2xl text-primary" />
        </div>
      ) : (
        <div className="flex flex-col gap-3">
          {articles.length === 0 && (
            <Card>
              <CardContent className="py-8 text-center text-sm text-muted-foreground">
                No articles in this list — try "Fetch new" or edit{" "}
                <code>hermes_cli/dashboard_config/articles.json</code>
              </CardContent>
            </Card>
          )}

          {articles.map((a) => (
            <Card key={a.id}>
              <CardContent className="flex items-start gap-4 py-4">
                <div className="min-w-0 flex-1">
                  <a
                    href={a.url ?? undefined}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-1.5 text-sm font-medium hover:underline"
                  >
                    <span className="truncate">{a.title}</span>
                    {a.url && (
                      <ExternalLink className="h-3 w-3 shrink-0 text-muted-foreground" />
                    )}
                  </a>
                  {a.authors.length > 0 && (
                    <div className="mt-1 truncate text-xs text-muted-foreground">
                      {a.authors.join(", ")}
                    </div>
                  )}
                  {a.abstract && (
                    <p className="mt-2 line-clamp-3 text-xs text-muted-foreground">
                      {a.abstract}
                    </p>
                  )}
                  <div className="mt-2 flex items-center gap-2 text-[11px] text-muted-foreground">
                    {a.publishedDate && <span>{a.publishedDate}</span>}
                    <Badge tone="outline">{a.source}</Badge>
                  </div>
                </div>

                <div className="flex shrink-0 items-center gap-1">
                  {status !== "saved" && (
                    <Button
                      ghost
                      size="icon"
                      title="Save"
                      aria-label="Save"
                      disabled={updating === a.id}
                      onClick={() => handleSetStatus(a.id, "saved")}
                    >
                      <BookMarked className="h-4 w-4" />
                    </Button>
                  )}
                  {status !== "read" && (
                    <Button
                      ghost
                      size="icon"
                      title="Mark read"
                      aria-label="Mark read"
                      disabled={updating === a.id}
                      onClick={() => handleSetStatus(a.id, "read")}
                    >
                      <Check className="h-4 w-4" />
                    </Button>
                  )}
                  {status !== "dismissed" && (
                    <Button
                      ghost
                      size="icon"
                      title="Dismiss"
                      aria-label="Dismiss"
                      className="text-destructive"
                      disabled={updating === a.id}
                      onClick={() => handleSetStatus(a.id, "dismissed")}
                    >
                      <X className="h-4 w-4" />
                    </Button>
                  )}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
