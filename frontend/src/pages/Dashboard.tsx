import { useState } from "react";
import Navbar from "@/components/layout/Navbar";
import Footer from "@/components/layout/Footer";
import UserStats from "@/components/dashboard/UserStats";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import { API } from "@/services/api";

const Dashboard = () => {
  const [user, setUser] = useState({
    name: "John Doe",
    email: "john@example.com",
    role: "user",
    avatar: "/placeholder.svg",
  });

  return (
    <div className="min-h-screen flex flex-col">
      <Navbar />
      
      <main className="flex-1 py-8">
        <div className="container mx-auto px-4">
          <div className="mb-8">
            <h1 className="text-3xl font-bold mb-2">Dashboard</h1>
            <p className="text-gray-500">View your prediction statistics and account information</p>
          </div>
          
          <div className="grid gap-8 md:grid-cols-[300px_1fr]">
            {/* Sidebar */}
            <div className="space-y-6">
              <Card>
                <CardContent className="pt-6">
                  <div className="flex flex-col items-center">
                    <Avatar className="h-24 w-24 mb-4">
                      <AvatarImage src={user.avatar} />
                      <AvatarFallback>{user.name.charAt(0)}</AvatarFallback>
                    </Avatar>
                    <h2 className="text-xl font-bold">{user.name}</h2>
                    <p className="text-sm text-gray-500">{user.email}</p>
                    <p className="text-xs bg-soccer-green/10 text-soccer-green px-2 py-1 rounded-full mt-2 uppercase">
                      {user.role}
                    </p>
                  </div>
                </CardContent>
              </Card>
              
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Account</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <Button variant="outline" className="w-full justify-start">
                    Profile Settings
                  </Button>
                  <Button variant="outline" className="w-full justify-start">
                    Notification Preferences
                  </Button>
                  <Button variant="outline" className="w-full justify-start">
                    Privacy Settings
                  </Button>
                  <Button variant="destructive" className="w-full justify-start">
                    Log Out
                  </Button>
                </CardContent>
              </Card>
            </div>
            
            {/* Main Content */}
            <div className="space-y-8">
              <Tabs defaultValue="stats">
                <TabsList>
                  <TabsTrigger value="stats">Prediction Stats</TabsTrigger>
                  <TabsTrigger value="history">Prediction History</TabsTrigger>
                  <TabsTrigger value="favorites">Favorite Teams</TabsTrigger>
                </TabsList>
                <TabsContent value="stats" className="mt-6">
                  <UserStats />
                </TabsContent>
                <TabsContent value="history" className="mt-6">
                  <Card>
                    <CardHeader>
                      <CardTitle>Prediction History</CardTitle>
                      <CardDescription>
                        View your past predictions and their outcomes
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="relative overflow-x-auto">
                        <table className="w-full text-sm text-left">
                          <thead className="text-xs uppercase bg-gray-50">
                            <tr>
                              <th scope="col" className="px-6 py-3">Match</th>
                              <th scope="col" className="px-6 py-3">Date</th>
                              <th scope="col" className="px-6 py-3">Your Prediction</th>
                              <th scope="col" className="px-6 py-3">Result</th>
                              <th scope="col" className="px-6 py-3">Status</th>
                            </tr>
                          </thead>
                          <tbody>
                            <tr className="bg-white border-b">
                              <td className="px-6 py-4">Arsenal vs Chelsea</td>
                              <td className="px-6 py-4">Apr 18, 2025</td>
                              <td className="px-6 py-4">Home Win</td>
                              <td className="px-6 py-4">2-1</td>
                              <td className="px-6 py-4">
                                <span className="px-2 py-1 text-xs rounded-full bg-soccer-win/10 text-soccer-win">Correct</span>
                              </td>
                            </tr>
                            <tr className="bg-gray-50 border-b">
                              <td className="px-6 py-4">Barcelona vs Real Madrid</td>
                              <td className="px-6 py-4">Apr 16, 2025</td>
                              <td className="px-6 py-4">Home Win</td>
                              <td className="px-6 py-4">1-3</td>
                              <td className="px-6 py-4">
                                <span className="px-2 py-1 text-xs rounded-full bg-soccer-loss/10 text-soccer-loss">Incorrect</span>
                              </td>
                            </tr>
                            <tr className="bg-white border-b">
                              <td className="px-6 py-4">Bayern vs Dortmund</td>
                              <td className="px-6 py-4">Apr 15, 2025</td>
                              <td className="px-6 py-4">Home Win</td>
                              <td className="px-6 py-4">3-0</td>
                              <td className="px-6 py-4">
                                <span className="px-2 py-1 text-xs rounded-full bg-soccer-win/10 text-soccer-win">Correct</span>
                              </td>
                            </tr>
                            <tr className="bg-gray-50 border-b">
                              <td className="px-6 py-4">PSG vs Lyon</td>
                              <td className="px-6 py-4">Apr 12, 2025</td>
                              <td className="px-6 py-4">Draw</td>
                              <td className="px-6 py-4">2-2</td>
                              <td className="px-6 py-4">
                                <span className="px-2 py-1 text-xs rounded-full bg-soccer-win/10 text-soccer-win">Correct</span>
                              </td>
                            </tr>
                            <tr className="bg-white">
                              <td className="px-6 py-4">Milan vs Juventus</td>
                              <td className="px-6 py-4">Apr 10, 2025</td>
                              <td className="px-6 py-4">Home Win</td>
                              <td className="px-6 py-4">1-0</td>
                              <td className="px-6 py-4">
                                <span className="px-2 py-1 text-xs rounded-full bg-soccer-win/10 text-soccer-win">Correct</span>
                              </td>
                            </tr>
                          </tbody>
                        </table>
                      </div>
                    </CardContent>
                  </Card>
                </TabsContent>
                <TabsContent value="favorites" className="mt-6">
                  <Card>
                    <CardHeader>
                      <CardTitle>Favorite Teams</CardTitle>
                      <CardDescription>
                        Teams you follow for predictions and updates
                      </CardDescription>
                    </CardHeader>
                    <CardContent className="text-center py-12">
                      <p className="text-muted-foreground mb-4">
                        You haven't added any favorite teams yet.
                      </p>
                      <Button className="bg-soccer-green hover:bg-soccer-green/90">
                        Add Favorite Teams
                      </Button>
                    </CardContent>
                  </Card>
                </TabsContent>
              </Tabs>
            </div>
          </div>
        </div>
      </main>
      
      <Footer />
    </div>
  );
};

export default Dashboard;
