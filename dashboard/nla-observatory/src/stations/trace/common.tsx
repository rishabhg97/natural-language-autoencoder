import { Badge } from "../../components/ui";

/**
 * Claim-scope badge that every TRACE panel carries: everything in this station
 * is a fresh extraction outside the qualified stored-snapshot channel claim.
 * The scope string always comes from the loaded shard, never a literal.
 */
export function FreshBadge(props: { scope: string }) {
  return (
    <Badge
      status="exploratory"
      label="fresh-forward"
      title={`claim scope: ${props.scope} — fresh extraction outside the qualified stored-snapshot channel claim`}
    />
  );
}
