
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs";
import {
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
} from "recharts";

// Mock data
const predictionAccuracyData = [
  { name: "Correct", value: 68 },
  { name: "Incorrect", value: 32 },
];

const leagueAccuracyData = [
  { name: "Premier League", correct: 78, incorrect: 22 },
  { name: "La Liga", correct: 65, incorrect: 35 },
  { name: "Bundesliga", correct: 72, incorrect: 28 },
  { name: "Serie A", correct: 70, incorrect: 30 },
  { name: "Ligue 1", correct: 62, incorrect: 38 },
];

const COLORS = ["#22C55E", "#EF4444"];

export default function UserStats() {
  return (
    <Tabs defaultValue="overview" className="space-y-4">
      <TabsList>
        <TabsTrigger value="overview">Overview</TabsTrigger>
        <TabsTrigger value="leagues">By League</TabsTrigger>
        <TabsTrigger value="teams">By Team</TabsTrigger>
      </TabsList>
      <TabsContent value="overview" className="space-y-4">
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">
                Total Predictions
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">156</div>
              <p className="text-xs text-muted-foreground">
                +23% from last month
              </p>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">
                Accuracy Rate
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">68%</div>
              <p className="text-xs text-muted-foreground">
                +5% from last month
              </p>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">
                Best League
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">Premier League</div>
              <p className="text-xs text-muted-foreground">
                78% prediction accuracy
              </p>
            </CardContent>
          </Card>
        </div>
        <div className="grid gap-4 md:grid-cols-2">
          <Card>
            <CardHeader>
              <CardTitle>Prediction Accuracy</CardTitle>
              <CardDescription>
                Overall accuracy of your predictions
              </CardDescription>
            </CardHeader>
            <CardContent className="flex justify-center">
              <div className="h-[300px] w-[300px]">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={predictionAccuracyData}
                      cx="50%"
                      cy="50%"
                      innerRadius={60}
                      outerRadius={80}
                      fill="#8884d8"
                      paddingAngle={5}
                      dataKey="value"
                      label={({ name, percent }) =>
                        `${name} ${(percent * 100).toFixed(0)}%`
                      }
                    >
                      {predictionAccuracyData.map((entry, index) => (
                        <Cell
                          key={`cell-${index}`}
                          fill={COLORS[index % COLORS.length]}
                        />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle>Recent Activity</CardTitle>
              <CardDescription>
                Your last 5 prediction results
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="flex items-center">
                  <div className="w-2 h-2 rounded-full bg-soccer-win mr-2"></div>
                  <div className="flex-1">
                    <div className="flex justify-between">
                      <p className="text-sm font-medium">Arsenal vs Chelsea</p>
                      <span className="text-sm text-muted-foreground">2 hours ago</span>
                    </div>
                    <p className="text-xs text-muted-foreground">Predicted: Home Win</p>
                  </div>
                </div>
                <div className="flex items-center">
                  <div className="w-2 h-2 rounded-full bg-soccer-loss mr-2"></div>
                  <div className="flex-1">
                    <div className="flex justify-between">
                      <p className="text-sm font-medium">Barcelona vs Real Madrid</p>
                      <span className="text-sm text-muted-foreground">1 day ago</span>
                    </div>
                    <p className="text-xs text-muted-foreground">Predicted: Away Win</p>
                  </div>
                </div>
                <div className="flex items-center">
                  <div className="w-2 h-2 rounded-full bg-soccer-win mr-2"></div>
                  <div className="flex-1">
                    <div className="flex justify-between">
                      <p className="text-sm font-medium">Bayern vs Dortmund</p>
                      <span className="text-sm text-muted-foreground">2 days ago</span>
                    </div>
                    <p className="text-xs text-muted-foreground">Predicted: Home Win</p>
                  </div>
                </div>
                <div className="flex items-center">
                  <div className="w-2 h-2 rounded-full bg-soccer-draw mr-2"></div>
                  <div className="flex-1">
                    <div className="flex justify-between">
                      <p className="text-sm font-medium">PSG vs Lyon</p>
                      <span className="text-sm text-muted-foreground">3 days ago</span>
                    </div>
                    <p className="text-xs text-muted-foreground">Predicted: Draw</p>
                  </div>
                </div>
                <div className="flex items-center">
                  <div className="w-2 h-2 rounded-full bg-soccer-win mr-2"></div>
                  <div className="flex-1">
                    <div className="flex justify-between">
                      <p className="text-sm font-medium">Milan vs Juventus</p>
                      <span className="text-sm text-muted-foreground">5 days ago</span>
                    </div>
                    <p className="text-xs text-muted-foreground">Predicted: Home Win</p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </TabsContent>
      <TabsContent value="leagues" className="space-y-4">
        <Card>
          <CardHeader>
            <CardTitle>League Performance</CardTitle>
            <CardDescription>
              Prediction accuracy by league
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-[300px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={leagueAccuracyData}
                  margin={{
                    top: 20,
                    right: 30,
                    left: 20,
                    bottom: 5,
                  }}
                >
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="correct" stackId="a" fill="#22C55E" name="Correct" />
                  <Bar dataKey="incorrect" stackId="a" fill="#EF4444" name="Incorrect" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </TabsContent>
      <TabsContent value="teams">
        <Card>
          <CardHeader>
            <CardTitle>Team Performance</CardTitle>
            <CardDescription>Coming soon</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="text-center py-12">
              <p className="text-muted-foreground">
                Team performance analytics will be available soon.
              </p>
            </div>
          </CardContent>
        </Card>
      </TabsContent>
    </Tabs>
  );
}
