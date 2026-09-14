import { useCallback, useEffect, useState } from "react";
import { RefreshCw, TrendingDown, TrendingUp, Wallet } from "lucide-react";
import { Button } from "@nous-research/ui/ui/components/button";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { H2 } from "@nous-research/ui/ui/components/typography/h2";
import { Card, CardContent } from "@nous-research/ui/ui/components/card";
import { useToast } from "@nous-research/ui/hooks/use-toast";
import { Toast } from "@nous-research/ui/ui/components/toast";
import { api } from "@/lib/api";
import type { PortfolioCrypto, PortfolioResponse, PortfolioStock } from "@/lib/api";
import { cn } from "@/lib/utils";

function formatPrice(price: number | null, currency: string | null): string {
  if (price === null) return "—";
  try {
    return new Intl.NumberFormat(undefined, {
      style: currency ? "currency" : "decimal",
      currency: currency ?? undefined,
      maximumFractionDigits: price < 1 ? 6 : 2,
    }).format(price);
  } catch {
    return `${price} ${currency ?? ""}`.trim();
  }
}

function PositionCard({
  label,
  ticker,
  price,
  currency,
}: {
  label: string;
  ticker: string;
  price: number | null;
  currency: string | null;
}) {
  const missing = price === null;
  return (
    <Card>
      <CardContent className="flex items-center justify-between gap-4 py-4">
        <div className="min-w-0">
          <div className="truncate text-sm font-medium">{label}</div>
          <div className="text-xs text-muted-foreground uppercase tracking-wide">
            {ticker}
          </div>
        </div>
        <div
          className={cn(
            "flex shrink-0 items-center gap-1.5 font-mono text-sm",
            missing && "text-muted-foreground",
          )}
        >
          {!missing &&
            (price >= 0 ? (
              <TrendingUp className="h-3.5 w-3.5 text-emerald-500" />
            ) : (
              <TrendingDown className="h-3.5 w-3.5 text-destructive" />
            ))}
          {formatPrice(price, currency)}
        </div>
      </CardContent>
    </Card>
  );
}

export default function PortfolioPage() {
  const [data, setData] = useState<PortfolioResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const { toast, showToast } = useToast();

  const load = useCallback(
    (silent = false) => {
      if (!silent) setRefreshing(true);
      return api
        .getPortfolio()
        .then((res) => setData(res))
        .catch(() => showToast("Failed to load portfolio", "error"))
        .finally(() => {
          setLoading(false);
          setRefreshing(false);
        });
    },
    [showToast],
  );

  useEffect(() => {
    load(true);
  }, [load]);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-24">
        <Spinner className="text-2xl text-primary" />
      </div>
    );
  }

  const stocks: PortfolioStock[] = data?.stocks ?? [];
  const crypto: PortfolioCrypto[] = data?.crypto ?? [];

  return (
    <div className="flex flex-col gap-6">
      <Toast toast={toast} />

      <div className="flex items-center justify-between">
        <H2 variant="sm" className="flex items-center gap-2 text-muted-foreground">
          <Wallet className="h-4 w-4" />
          Portfolio
        </H2>
        <Button
          size="sm"
          ghost
          onClick={() => load()}
          disabled={refreshing}
          prefix={
            refreshing ? (
              <Spinner />
            ) : (
              <RefreshCw className="h-3.5 w-3.5" />
            )
          }
        >
          Refresh
        </Button>
      </div>

      <div className="flex flex-col gap-3">
        <div className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
          Stocks ({stocks.length})
        </div>
        {stocks.length === 0 && (
          <Card>
            <CardContent className="py-8 text-center text-sm text-muted-foreground">
              No stock positions configured — edit{" "}
              <code>hermes_cli/dashboard_config/portfolio.json</code>
            </CardContent>
          </Card>
        )}
        {stocks.map((s) => (
          <PositionCard
            key={s.ticker}
            label={s.label ?? s.ticker}
            ticker={s.ticker}
            price={s.price}
            currency={s.currency}
          />
        ))}
      </div>

      <div className="flex flex-col gap-3">
        <div className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
          Crypto ({crypto.length})
        </div>
        {crypto.length === 0 && (
          <Card>
            <CardContent className="py-8 text-center text-sm text-muted-foreground">
              No crypto positions configured
            </CardContent>
          </Card>
        )}
        {crypto.map((c) => (
          <PositionCard
            key={c.id}
            label={c.label ?? c.id}
            ticker={c.id}
            price={c.price}
            currency={c.currency}
          />
        ))}
      </div>
    </div>
  );
}
